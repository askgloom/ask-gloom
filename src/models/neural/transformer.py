import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class TransformerConfig:
    input_size: int
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_sequence_length: int = 1000
    positional_encoding: bool = True

@dataclass
class TransformerOutput:
    predictions: torch.Tensor
    attention_maps: Dict[str, torch.Tensor]
    confidence: float
    metadata: Dict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class GloomTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        
        # Positional encoding
        if config.positional_encoding:
            self.pos_encoder = PositionalEncoding(config.d_model, config.max_sequence_length)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.input_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.d_model)

class TransformerAnalyzer:
    def __init__(self, config: Optional[TransformerConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize configuration
        self.config = config or TransformerConfig(input_size=1)
        
        # Initialize model
        self.model = GloomTransformer(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.0001, 
            betas=(0.9, 0.98), 
            eps=1e-9
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Analysis state
        self.training_history: List[Dict] = []
        self.attention_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def _create_mask(self, size: int) -> torch.Tensor:
        """Create attention mask for sequence"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(self.device)

    def _prepare_data(self, 
                     data: Union[np.ndarray, torch.Tensor], 
                     sequence_length: Optional[int] = None) -> torch.Tensor:
        """Prepare data for transformer processing"""
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        if sequence_length is None:
            sequence_length = min(len(data), self.config.max_sequence_length)
            
        # Reshape and pad if necessary
        if len(data.shape) == 1:
            data = data.unsqueeze(-1)
            
        if len(data) > sequence_length:
            data = data[:sequence_length]
        elif len(data) < sequence_length:
            pad_length = sequence_length - len(data)
            data = torch.pad(data, (0, 0, 0, pad_length))
            
        return data.to(self.device)

    async def train(self, 
                   data: Union[np.ndarray, torch.Tensor], 
                   epochs: int = 100, 
                   batch_size: int = 32) -> Dict:
        """Train the transformer model"""
        self.model.train()
        training_stats = []
        
        try:
            sequences = self._prepare_data(data)
            src_mask = self._create_mask(sequences.size(1))
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for i in range(0, len(sequences), batch_size):
                    batch = sequences[i:i + batch_size]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.model(batch, src_mask=src_mask)
                    loss = self.criterion(output, batch)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Record training stats
                stats = {
                    'epoch': epoch,
                    'loss': epoch_loss / len(sequences),
                    'timestamp': datetime.now()
                }
                training_stats.append(stats)
                
                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}: Loss = {stats['loss']:.6f}")
            
            self.training_history.extend(training_stats)
            return {'status': 'success', 'history': training_stats}
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise TrainingError(f"Transformer training failed: {str(e)}")

    async def analyze(self, 
                     data: Union[np.ndarray, torch.Tensor], 
                     store_attention: bool = True) -> TransformerOutput:
        """Analyze patterns using the transformer model"""
        self.model.eval()
        
        try:
            with torch.no_grad():
                sequences = self._prepare_data(data)
                src_mask = self._create_mask(sequences.size(1))
                
                # Get model outputs and attention maps
                outputs = self.model(sequences, src_mask=src_mask)
                attention_maps = self.model.transformer.encoder.layers[-1].self_attn.attention_weights
                
                # Calculate prediction confidence
                prediction_std = torch.std(outputs, dim=1)
                confidence = 1.0 / (1.0 + prediction_std.mean().item())
                
                # Store attention maps if requested
                if store_attention:
                    attention_id = f"attn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.attention_cache[attention_id] = attention_maps
                
                return TransformerOutput(
                    predictions=outputs.cpu(),
                    attention_maps={k: v.cpu() for k, v in attention_maps.items()},
                    confidence=float(confidence),
                    metadata={
                        'timestamp': datetime.now(),
                        'sequence_length': sequences.size(1),
                        'attention_id': attention_id if store_attention else None
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"Transformer analysis failed: {str(e)}")

    def get_attention_maps(self, attention_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """Retrieve stored attention maps"""
        return self.attention_cache.get(attention_id)

class TrainingError(Exception):
    pass

class AnalysisError(Exception):
    pass

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create sample data
        data = np.sin(np.linspace(0, 100, 1000)).reshape(-1, 1)
        
        # Initialize analyzer
        config = TransformerConfig(input_size=1)
        analyzer = TransformerAnalyzer(config)
        
        # Train model
        await analyzer.train(data, epochs=50)
        
        # Analyze patterns
        output = await analyzer.analyze(data)
        print(f"Analysis confidence: {output.confidence:.4f}")
        print(f"Attention maps stored: {list(output.attention_maps.keys())}")

    import asyncio
    asyncio.run(main())
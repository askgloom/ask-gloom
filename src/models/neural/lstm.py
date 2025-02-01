import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class LSTMConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    batch_first: bool = True
    sequence_length: int = 50

@dataclass
class LSTMPrediction:
    values: torch.Tensor
    confidence: float
    attention_weights: Optional[torch.Tensor] = None
    metadata: Dict = None

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended, attention_weights

class GloomLSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=config.batch_first
        )
        
        # Attention mechanism
        lstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.attention = AttentionLayer(lstm_output_size)
        
        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, config.input_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

class LSTMAnalyzer:
    def __init__(self, config: Optional[LSTMConfig] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize configuration
        self.config = config or LSTMConfig(input_size=1)
        
        # Initialize model
        self.model = GloomLSTM(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
        # Analysis state
        self.training_history: List[Dict] = []
        self.prediction_cache: Dict[str, LSTMPrediction] = {}

    def _prepare_sequence(self, 
                         data: Union[np.ndarray, torch.Tensor], 
                         sequence_length: Optional[int] = None) -> torch.Tensor:
        """Prepare data sequences for LSTM processing"""
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        seq_length = sequence_length or self.config.sequence_length
        sequences = []
        
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i + seq_length])
            
        return torch.stack(sequences).to(self.device)

    async def train(self, 
                   data: Union[np.ndarray, torch.Tensor], 
                   epochs: int = 100, 
                   batch_size: int = 32) -> Dict:
        """Train the LSTM model on provided data"""
        self.model.train()
        sequences = self._prepare_sequence(data)
        
        training_stats = []
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                # Process in batches
                for i in range(0, len(sequences), batch_size):
                    batch = sequences[i:i + batch_size]
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(batch)
                    loss = self.criterion(outputs, batch)
                    
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
            raise TrainingError(f"LSTM training failed: {str(e)}")

    async def analyze(self, 
                     data: Union[np.ndarray, torch.Tensor], 
                     return_attention: bool = True) -> LSTMPrediction:
        """Analyze patterns in the data using the LSTM model"""
        self.model.eval()
        
        try:
            with torch.no_grad():
                sequences = self._prepare_sequence(data)
                
                # Get model outputs and attention weights
                outputs, attention = self.model.attention(sequences)
                
                # Calculate prediction confidence
                prediction_std = torch.std(outputs, dim=0)
                confidence = 1.0 / (1.0 + prediction_std.mean().item())
                
                prediction = LSTMPrediction(
                    values=outputs.cpu(),
                    confidence=float(confidence),
                    attention_weights=attention.cpu() if return_attention else None,
                    metadata={
                        'timestamp': datetime.now(),
                        'sequence_length': self.config.sequence_length
                    }
                )
                
                # Cache prediction
                cache_key = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.prediction_cache[cache_key] = prediction
                
                return prediction
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"LSTM analysis failed: {str(e)}")

    def get_cached_prediction(self, prediction_id: str) -> Optional[LSTMPrediction]:
        """Retrieve a cached prediction"""
        return self.prediction_cache.get(prediction_id)

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
        config = LSTMConfig(input_size=1)
        analyzer = LSTMAnalyzer(config)
        
        # Train model
        await analyzer.train(data, epochs=50)
        
        # Analyze patterns
        prediction = await analyzer.analyze(data)
        print(f"Prediction confidence: {prediction.confidence:.4f}")

    import asyncio
    asyncio.run(main())
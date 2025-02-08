"""
Core text processing module for Ask-Gloom.
Handles text understanding, tokenization, and semantic analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from ask_gloom.utils.types import (
    ProcessorConfig,
    ProcessedText,
    EntityInfo,
    SemanticInfo
)
from ask_gloom.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextProcessor:
    """
    Handles text processing and understanding for Ask-Gloom.
    Provides tokenization, entity recognition, and semantic analysis.
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize text processor with configuration.
        
        Args:
            config: Processor configuration settings
        """
        self.config = config or ProcessorConfig()
        
        # Initialize models
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name
        )
        
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        logger.info(f"Initialized TextProcessor with model: {self.config.model_name}")

    def process(self, text: str) -> ProcessedText:
        """
        Process input text through the full pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            ProcessedText object containing analysis results
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Tokenize and get embeddings
        embeddings = self._get_embeddings(text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Analyze semantics
        semantics = self._analyze_semantics(text, embeddings)
        
        # Create processed result
        result = ProcessedText(
            original_text=text,
            embeddings=embeddings,
            entities=entities,
            semantics=semantics,
            metadata={
                "model": self.config.model_name,
                "version": self.config.version
            }
        )
        
        logger.debug(f"Processed text: {text[:50]}...")
        return result

    def _get_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for input text.
        
        Args:
            text: Input text
            
        Returns:
            Text embeddings as numpy array
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.cpu().numpy()

    def _extract_entities(self, text: str) -> List[EntityInfo]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Basic entity extraction (placeholder for more sophisticated implementation)
        words = text.split()
        for word in words:
            if word[0].isupper():
                entities.append(EntityInfo(
                    text=word,
                    type="UNKNOWN",
                    start=text.index(word),
                    end=text.index(word) + len(word),
                    confidence=0.5
                ))
        
        return entities

    def _analyze_semantics(
        self,
        text: str,
        embeddings: np.ndarray
    ) -> SemanticInfo:
        """
        Analyze semantic content of text.
        
        Args:
            text: Input text
            embeddings: Text embeddings
            
        Returns:
            Semantic analysis results
        """
        # Placeholder for more sophisticated semantic analysis
        sentiment_score = 0.0
        if any(word in text.lower() for word in ["good", "great", "happy"]):
            sentiment_score = 0.8
        elif any(word in text.lower() for word in ["bad", "awful", "sad"]):
            sentiment_score = 0.2
            
        return SemanticInfo(
            sentiment=sentiment_score,
            topics=[],  # Placeholder for topic extraction
            intent="unknown",  # Placeholder for intent classification
            confidence=0.5
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words/subwords.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return self.tokenizer.tokenize(text)

    def batch_process(
        self,
        texts: List[str]
    ) -> List[ProcessedText]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed results
        """
        return [self.process(text) for text in texts]

    def save_state(self, path: str) -> None:
        """
        Save processor state to disk.
        
        Args:
            path: Path to save directory
        """
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        logger.info(f"Saved processor state to {path}")

    def load_state(self, path: str) -> None:
        """
        Load processor state from disk.
        
        Args:
            path: Path to load directory
        """
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.to(self.device)
        logger.info(f"Loaded processor state from {path}")

    @property
    def model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        return {
            "name": self.config.model_name,
            "version": self.config.version,
            "device": str(self.device)
        }
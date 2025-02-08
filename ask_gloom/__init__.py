"""
Ask-Gloom: A conversational AI framework powered by cognitive architecture.

This module provides the main interface for the Ask-Gloom framework,
integrating natural language processing with Gloom's cognitive systems.
"""

from typing import Dict, List, Optional, Tuple, Union
import logging

from ask_gloom.core.conversation import ConversationManager
from ask_gloom.core.processor import TextProcessor
from ask_gloom.models.personality import PersonalityModel
from ask_gloom.memory.episodic import EpisodicMemory
from ask_gloom.memory.semantic import SemanticMemory
from ask_gloom.utils.logger import setup_logger
from ask_gloom.utils.types import (
    ConversationContext,
    MemoryConfig,
    ProcessorConfig,
    Response
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Setup logging
logger = setup_logger(__name__)

class AskGloom:
    """
    Main interface for the Ask-Gloom conversational AI system.
    
    Attributes:
        conversation_manager: Manages conversation flow and context
        processor: Handles text processing and understanding
        personality: Manages AI personality and response style
        episodic_memory: Stores conversation history
        semantic_memory: Manages conceptual knowledge
    """
    
    def __init__(
        self,
        memory_config: Optional[MemoryConfig] = None,
        processor_config: Optional[ProcessorConfig] = None,
        personality_path: Optional[str] = None
    ):
        """
        Initialize AskGloom with specified configurations.
        
        Args:
            memory_config: Configuration for memory systems
            processor_config: Configuration for text processing
            personality_path: Path to personality model file
        """
        logger.info("Initializing AskGloom...")
        
        # Initialize components
        self.conversation_manager = ConversationManager()
        self.processor = TextProcessor(config=processor_config)
        self.personality = PersonalityModel(model_path=personality_path)
        
        # Setup memory systems
        memory_config = memory_config or MemoryConfig()
        self.episodic_memory = EpisodicMemory(config=memory_config)
        self.semantic_memory = SemanticMemory(config=memory_config)
        
        logger.info("AskGloom initialization complete")

    def process(
        self,
        text: str,
        context: Optional[ConversationContext] = None
    ) -> Response:
        """
        Process input text and generate a response.
        
        Args:
            text: Input text to process
            context: Optional conversation context
            
        Returns:
            Response object containing generated response and metadata
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        logger.debug(f"Processing input: {text[:50]}...")
        
        # Process input through pipeline
        processed_input = self.processor.process(text)
        conversation_context = self.conversation_manager.get_context(context)
        
        # Query memories
        episodic_context = self.episodic_memory.query_relevant(
            text,
            conversation_context
        )
        semantic_context = self.semantic_memory.query_relevant(
            text,
            conversation_context
        )
        
        # Generate response
        response = self.personality.generate_response(
            processed_input,
            conversation_context,
            episodic_context,
            semantic_context
        )
        
        # Update memories
        self.episodic_memory.store(text, response, conversation_context)
        self.semantic_memory.update(text, response, conversation_context)
        
        logger.debug(f"Generated response: {response.text[:50]}...")
        return response

    def reset_conversation(self) -> None:
        """Reset the current conversation context."""
        self.conversation_manager.reset()
        logger.info("Conversation context reset")

    def save_state(self, path: str) -> None:
        """
        Save the current state to disk.
        
        Args:
            path: Directory path to save state
        """
        logger.info(f"Saving state to {path}")
        self.conversation_manager.save(f"{path}/conversation.pkl")
        self.episodic_memory.save(f"{path}/episodic.pkl")
        self.semantic_memory.save(f"{path}/semantic.pkl")

    def load_state(self, path: str) -> None:
        """
        Load state from disk.
        
        Args:
            path: Directory path to load state from
        """
        logger.info(f"Loading state from {path}")
        self.conversation_manager.load(f"{path}/conversation.pkl")
        self.episodic_memory.load(f"{path}/episodic.pkl")
        self.semantic_memory.load(f"{path}/semantic.pkl")

    @property
    def version(self) -> str:
        """Get the current version of Ask-Gloom."""
        return __version__

# Convenience functions
def create_agent(**kwargs) -> AskGloom:
    """Create a new AskGloom agent with default settings."""
    return AskGloom(**kwargs)

def load_agent(path: str) -> AskGloom:
    """Load an AskGloom agent from disk."""
    agent = AskGloom()
    agent.load_state(path)
    return agent

# Version information
__all__ = [
    'AskGloom',
    'create_agent',
    'load_agent',
    'ConversationContext',
    'MemoryConfig',
    'ProcessorConfig',
    'Response'
]
"""
Core conversation management module for Ask-Gloom.
Handles conversation flow, context tracking, and state management.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
import uuid

from ask_gloom.utils.types import ConversationContext, Message
from ask_gloom.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Conversation:
    """
    Represents a single conversation session.
    
    Attributes:
        id: Unique conversation identifier
        messages: List of messages in the conversation
        context: Additional conversation context
        metadata: Conversation metadata
        created_at: Conversation creation timestamp
        updated_at: Last update timestamp
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class ConversationManager:
    """
    Manages conversation sessions and their associated contexts.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize the conversation manager.
        
        Args:
            max_history: Maximum number of messages to retain per conversation
        """
        self.conversations: Dict[str, Conversation] = {}
        self.max_history = max_history
        self.current_conversation: Optional[str] = None
        logger.info("Conversation manager initialized")

    def create_conversation(
        self,
        context: Optional[ConversationContext] = None
    ) -> str:
        """
        Create a new conversation session.
        
        Args:
            context: Initial conversation context
            
        Returns:
            Conversation ID
        """
        conversation = Conversation()
        if context:
            conversation.context.update(context)
        
        self.conversations[conversation.id] = conversation
        self.current_conversation = conversation.id
        
        logger.debug(f"Created new conversation: {conversation.id}")
        return conversation.id

    def add_message(
        self,
        text: str,
        role: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a message to a conversation.
        
        Args:
            text: Message content
            role: Message sender role (user/assistant)
            conversation_id: Target conversation ID
            metadata: Additional message metadata
            
        Raises:
            ValueError: If conversation_id is invalid
        """
        conv_id = conversation_id or self.current_conversation
        if not conv_id or conv_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        conversation = self.conversations[conv_id]
        message = Message(
            text=text,
            role=role,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
        
        # Prune history if needed
        if len(conversation.messages) > self.max_history:
            conversation.messages = conversation.messages[-self.max_history:]
        
        logger.debug(f"Added message to conversation {conv_id}")

    def get_context(
        self,
        conversation_id: Optional[str] = None,
        window_size: int = 5
    ) -> ConversationContext:
        """
        Get recent conversation context.
        
        Args:
            conversation_id: Target conversation ID
            window_size: Number of recent messages to include
            
        Returns:
            Conversation context including recent history
            
        Raises:
            ValueError: If conversation_id is invalid
        """
        conv_id = conversation_id or self.current_conversation
        if not conv_id or conv_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        conversation = self.conversations[conv_id]
        recent_messages = conversation.messages[-window_size:]
        
        return {
            "conversation_id": conv_id,
            "history": recent_messages,
            "context": conversation.context.copy(),
            "metadata": conversation.metadata.copy()
        }

    def update_context(
        self,
        context: Dict[str, str],
        conversation_id: Optional[str] = None
    ) -> None:
        """
        Update conversation context.
        
        Args:
            context: New context values
            conversation_id: Target conversation ID
            
        Raises:
            ValueError: If conversation_id is invalid
        """
        conv_id = conversation_id or self.current_conversation
        if not conv_id or conv_id not in self.conversations:
            raise ValueError("Invalid conversation ID")
            
        self.conversations[conv_id].context.update(context)
        self.conversations[conv_id].updated_at = datetime.now()
        
        logger.debug(f"Updated context for conversation {conv_id}")

    def save(self, path: str) -> None:
        """
        Save conversation state to disk.
        
        Args:
            path: Path to save file
        """
        state = {
            "conversations": self.conversations,
            "current_conversation": self.current_conversation,
            "max_history": self.max_history
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved conversation state to {path}")

    def load(self, path: str) -> None:
        """
        Load conversation state from disk.
        
        Args:
            path: Path to load file
            
        Raises:
            FileNotFoundError: If save file doesn't exist
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.conversations = state["conversations"]
        self.current_conversation = state["current_conversation"]
        self.max_history = state["max_history"]
        
        logger.info(f"Loaded conversation state from {path}")

    def reset(self, conversation_id: Optional[str] = None) -> None:
        """
        Reset conversation history.
        
        Args:
            conversation_id: Target conversation ID
        """
        conv_id = conversation_id or self.current_conversation
        if conv_id and conv_id in self.conversations:
            self.conversations[conv_id].messages.clear()
            self.conversations[conv_id].updated_at = datetime.now()
            logger.info(f"Reset conversation {conv_id}")
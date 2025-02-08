"""
Core memory interface for Ask-Gloom.
Provides base memory functionality and interfaces for memory systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import pickle
from dataclasses import dataclass, field

from ask_gloom.utils.types import MemoryConfig, MemoryQuery, MemoryItem
from ask_gloom.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class BaseMemory:
    """
    Base memory structure for all memory types.
    
    Attributes:
        content: Main memory content
        timestamp: Creation time
        importance: Memory importance score (0-1)
        metadata: Additional memory metadata
        access_count: Number of times memory was accessed
        last_accessed: Last access timestamp
    """
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

class MemoryInterface(ABC):
    """
    Abstract base class defining the memory system interface.
    All memory implementations must inherit from this class.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory system with configuration.
        
        Args:
            config: Memory system configuration
        """
        self.config = config or MemoryConfig()
        self._memories: Dict[str, BaseMemory] = {}
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def store(self, item: MemoryItem) -> str:
        """
        Store a new memory item.
        
        Args:
            item: Memory item to store
            
        Returns:
            Memory ID
        """
        pass

    @abstractmethod
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID.
        
        Args:
            memory_id: ID of memory to retrieve
            
        Returns:
            Memory item if found, None otherwise
        """
        pass

    @abstractmethod
    def search(self, query: MemoryQuery) -> List[MemoryItem]:
        """
        Search memories based on query.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of matching memory items
        """
        pass

    def update_importance(self, memory_id: str, importance: float) -> bool:
        """
        Update importance score of a memory.
        
        Args:
            memory_id: Target memory ID
            importance: New importance score (0-1)
            
        Returns:
            True if update successful, False otherwise
        """
        if memory_id in self._memories:
            self._memories[memory_id].importance = max(0.0, min(1.0, importance))
            self._memories[memory_id].last_accessed = datetime.now()
            return True
        return False

    def prune(self, threshold: float = 0.2) -> int:
        """
        Remove low-importance memories.
        
        Args:
            threshold: Minimum importance threshold
            
        Returns:
            Number of memories removed
        """
        initial_count = len(self._memories)
        self._memories = {
            k: v for k, v in self._memories.items()
            if v.importance >= threshold
        }
        removed = initial_count - len(self._memories)
        logger.info(f"Pruned {removed} memories below threshold {threshold}")
        return removed

    def calculate_retention_score(self, memory: BaseMemory) -> float:
        """
        Calculate retention score for memory pruning decisions.
        
        Args:
            memory: Memory to evaluate
            
        Returns:
            Retention score (0-1)
        """
        now = datetime.now()
        age = (now - memory.timestamp).total_seconds()
        last_access = (now - memory.last_accessed).total_seconds()
        
        # Factors in retention calculation:
        # - Age (newer memories kept)
        # - Access frequency (frequently accessed memories kept)
        # - Importance (important memories kept)
        # - Recency of access (recently accessed memories kept)
        
        age_factor = 1.0 / (1.0 + age / 86400)  # Age in days
        access_factor = min(1.0, memory.access_count / 10.0)
        recency_factor = 1.0 / (1.0 + last_access / 3600)  # Last access in hours
        
        score = (
            0.3 * age_factor +
            0.2 * access_factor +
            0.3 * memory.importance +
            0.2 * recency_factor
        )
        
        return max(0.0, min(1.0, score))

    def save(self, path: str) -> None:
        """
        Save memory state to disk.
        
        Args:
            path: Path to save file
        """
        state = {
            "memories": self._memories,
            "config": self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved memory state to {path}")

    def load(self, path: str) -> None:
        """
        Load memory state from disk.
        
        Args:
            path: Path to load file
            
        Raises:
            FileNotFoundError: If save file doesn't exist
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self._memories = state["memories"]
        self.config = state["config"]
        
        logger.info(f"Loaded memory state from {path}")

    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()
        logger.info("Cleared all memories")

    @property
    def size(self) -> int:
        """Get number of stored memories."""
        return len(self._memories)

    def __len__(self) -> int:
        return self.size
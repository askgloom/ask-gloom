"""
Integration module for Ask Gloom.

This module handles various platform integrations like Telegram, Discord, etc.
Each integration should implement a common interface for consistency.
"""

from abc import ABC, abstractmethod

class Integration(ABC):
    """Base class for all integrations."""
    
    @abstractmethod
    def connect(self):
        """Establish connection with the platform."""
        pass
    
    @abstractmethod
    def send_message(self, message: str):
        """Send a message through the integration."""
        pass
    
    @abstractmethod
    def receive_message(self):
        """Receive messages from the integration."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Close the connection with the platform."""
        pass
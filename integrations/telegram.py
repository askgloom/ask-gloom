"""
Telegram integration for Ask Gloom.
Handles all Telegram-specific functionality and implements the Integration interface.
"""

import logging
from typing import Optional, Callable
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from . import Integration

logger = logging.getLogger(__name__)

class TelegramIntegration(Integration):
    def __init__(self, token: str, message_handler: Optional[Callable] = None):
        """
        Initialize Telegram integration.
        
        Args:
            token (str): Telegram bot token
            message_handler (Callable, optional): Custom message handling function
        """
        self.token = token
        self.application = None
        self.message_handler = message_handler or self._default_message_handler

    async def connect(self):
        """Establish connection with Telegram."""
        try:
            self.application = Application.builder().token(self.token).build()
            
            # Register handlers
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
            
            # Start the bot
            await self.application.initialize()
            await self.application.start()
            logger.info("Telegram bot connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect Telegram bot: {e}")
            raise

    async def send_message(self, message: str, chat_id: int):
        """
        Send a message through Telegram.
        
        Args:
            message (str): Message to send
            chat_id (int): Telegram chat ID to send message to
        """
        if not self.application:
            raise RuntimeError("Telegram bot not connected")
        
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def receive_message(self):
        """
        Receive messages from Telegram.
        This is handled automatically through the application's event loop.
        """
        pass

    async def disconnect(self):
        """Close the connection with Telegram."""
        if self.application:
            try:
                await self.application.stop()
                await self.application.shutdown()
                logger.info("Telegram bot disconnected successfully")
            except Exception as e:
                logger.error(f"Failed to disconnect Telegram bot: {e}")
                raise

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        welcome_message = (
            "👋 Welcome to Ask Gloom!\n\n"
            "I'm here to help you with browser automation tasks. "
            "Just send me your requests, and I'll handle them for you."
        )
        await update.message.reply_text(welcome_message)

    async def _default_message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Default handler for text messages."""
        await update.message.reply_text(
            "I received your message. However, no custom handler is configured."
        )
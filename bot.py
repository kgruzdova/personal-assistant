"""
Main Bot Module.
Initializes and configures the Telegram bot using pyTelegramBotAPI.
"""

from telebot.async_telebot import AsyncTeleBot

from config import TELEGRAM_BOT_TOKEN
from utils.logging import logger


# Create bot instance
bot = AsyncTeleBot(TELEGRAM_BOT_TOKEN, parse_mode='Markdown')

logger.info("Bot instance created")

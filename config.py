"""
Configuration module for the Personal Assistant Telegram Bot.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from dotenv import dotenv_values, load_dotenv

# Base directory
BASE_DIR = Path(__file__).parent
ENV_FILE = BASE_DIR / ".env"

# Read credentials from this project's .env, regardless of the working directory
# or identically named variables in the process environment.
env_values = dotenv_values(ENV_FILE)
load_dotenv(ENV_FILE, override=True)

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = env_values.get("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in .env file")

# OpenAI Configuration
OPENAI_API_KEY = env_values.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in .env file")

# All OpenAI requests use the official API directly.
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Bot Modes
class BotMode:
    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    RAG = "rag"

DEFAULT_MODE = os.getenv("BOT_MODE", BotMode.TEXT)

# Voice Configuration
class VoiceType:
    ALLOY = "alloy"      # Neutral
    ECHO = "echo"        # Male
    NOVA = "nova"        # Female
    FABLE = "fable"      # Male (British)
    ONYX = "onyx"        # Male (Deep)
    SHIMMER = "shimmer"  # Female (Warm)

DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", VoiceType.ALLOY)

# OpenAI Models
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GPT_MINI_MODEL = "gpt-4o-mini"
WHISPER_MODEL = "whisper-1"
TTS_MODEL = "tts-1"
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_DEFAULT_SIZE = os.getenv("OPENAI_IMAGE_SIZE", "1024x1024")
IMAGE_DEFAULT_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")

# Image configuration aliases
# Compatibility aliases for examples using the old setting names.
DALLE_MODEL = IMAGE_MODEL
DALLE_DEFAULT_SIZE = IMAGE_DEFAULT_SIZE
DALLE_DEFAULT_QUALITY = IMAGE_DEFAULT_QUALITY
DALLE_DEFAULT_STYLE = "vivid"  # Options: vivid, natural

# Database Configuration
DB_PATH = BASE_DIR / os.getenv("DB_PATH", "data/embeddings.db")

# Data paths
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EMBEDDINGS_DB = DATA_DIR / "embeddings.db"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
DOCUMENTS_DIR.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "bot.log"

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 3
PDF_OCR_LANGUAGE = os.getenv("PDF_OCR_LANGUAGE", "ru")

# OpenAI Settings
TEMPERATURE = 0.7
MAX_TOKENS = 1500

# User session settings
MAX_HISTORY_LENGTH = 10  # Maximum number of messages to keep in history


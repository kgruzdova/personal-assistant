"""
Helper functions for the Personal Assistant Bot.
Provides utility functions for file operations, audio conversion, etc.
"""

import os
import uuid
import aiofiles
from pathlib import Path
from typing import Optional, Union

from config import BASE_DIR
from utils.logging import logger


async def save_file_async(file_content: bytes, extension: str = "tmp") -> Path:
    """
    Save file content asynchronously to a temporary file.
    
    Args:
        file_content: Binary content of the file
        extension: File extension (without dot)
    
    Returns:
        Path to the saved file
    """
    filename = f"{uuid.uuid4()}.{extension}"
    filepath = BASE_DIR / "data" / filename
    
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(file_content)
        logger.debug(f"File saved: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise


def convert_ogg_to_wav(ogg_path: Union[str, Path]) -> Path:
    """
    Convert OGG audio file to WAV format using pydub.
    
    Args:
        ogg_path: Path to the OGG file
    
    Returns:
        Path to the converted WAV file
    """
    try:
        # Lazy import to avoid audioop error on startup
        from pydub import AudioSegment
    except ImportError as e:
        logger.error(f"pydub not available: {e}. Install ffmpeg and audioop support.")
        raise
    
    ogg_path = Path(ogg_path)
    wav_path = ogg_path.with_suffix('.wav')
    
    try:
        audio = AudioSegment.from_ogg(ogg_path)
        audio.export(wav_path, format='wav')
        logger.debug(f"Converted {ogg_path} to {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise


def cleanup_file(filepath: Union[str, Path]) -> None:
    """
    Delete a file safely.
    
    Args:
        filepath: Path to the file to delete
    """
    try:
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.warning(f"Error cleaning up file {filepath}: {e}")


def cleanup_files(*filepaths: Union[str, Path]) -> None:
    """
    Delete multiple files safely.
    
    Args:
        *filepaths: Paths to files to delete
    """
    for filepath in filepaths:
        cleanup_file(filepath)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


class UserSession:
    """Simple user session manager to store conversation history."""
    
    def __init__(self):
        self.sessions = {}
    
    def get_history(self, user_id: int) -> list:
        """Get conversation history for a user."""
        return self.sessions.get(user_id, [])
    
    def add_message(self, user_id: int, role: str, content: str):
        """Add a message to user's conversation history."""
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        
        self.sessions[user_id].append({
            "role": role,
            "content": content
        })
        
        # Limit history length
        from config import MAX_HISTORY_LENGTH
        if len(self.sessions[user_id]) > MAX_HISTORY_LENGTH * 2:
            self.sessions[user_id] = self.sessions[user_id][-MAX_HISTORY_LENGTH * 2:]
    
    def clear_history(self, user_id: int):
        """Clear conversation history for a user."""
        if user_id in self.sessions:
            del self.sessions[user_id]
    
    def get_mode(self, user_id: int) -> str:
        """Get current mode for a user."""
        return self.sessions.get(f"{user_id}_mode", "text")
    
    def set_mode(self, user_id: int, mode: str):
        """Set mode for a user."""
        self.sessions[f"{user_id}_mode"] = mode
    
    def get_voice(self, user_id: int) -> str:
        """Get current voice setting for a user."""
        from config import DEFAULT_VOICE
        return self.sessions.get(f"{user_id}_voice", DEFAULT_VOICE)
    
    def set_voice(self, user_id: int, voice: str):
        """Set voice for a user."""
        self.sessions[f"{user_id}_voice"] = voice


# Global session manager instance
user_sessions = UserSession()


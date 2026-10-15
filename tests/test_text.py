"""
Tests for text processing functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from services.openai_client import OpenAIClient
from services.router import route_text_request
from utils.helpers import UserSession


class TestTextProcessing:
    """Test suite for text processing."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        with patch('services.openai_client.AsyncOpenAI') as mock:
            mock_instance = Mock()
            mock_instance.chat = Mock()
            mock_instance.chat.completions = Mock()
            mock_instance.chat.completions.create = AsyncMock(
                return_value=Mock(
                    choices=[
                        Mock(message=Mock(content="Test response"))
                    ]
                )
            )
            mock.return_value = mock_instance
            yield mock
    
    @pytest.mark.asyncio
    async def test_generate_text_response(self, mock_openai_client):
        """Test basic text response generation."""
        client = OpenAIClient()
        
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        response = await client.generate_text_response(messages)
        
        assert response == "Test response"
        assert isinstance(response, str)
    
    @pytest.mark.asyncio
    async def test_route_text_request(self):
        """A regular text message needs only one OpenAI request."""
        with patch('services.router.openai_client.generate_reply_or_image', new_callable=AsyncMock) as generate:
            generate.return_value = {"text": "Test response"}
            response = await route_text_request(987654321, "Привет, что ты умеешь?", mode="text")

        assert response["text"] == "Test response"
        generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_route_text_reports_invalid_key(self):
        """A rejected key returns a useful message to the Telegram handler."""
        import httpx
        from openai import AuthenticationError

        response = httpx.Response(
            401,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        error = AuthenticationError("Unauthorized", response=response, body=None)
        with patch('services.router.openai_client.generate_reply_or_image', new_callable=AsyncMock) as generate:
            generate.side_effect = error
            result = await route_text_request(987654322, "Привет", mode="text")

        assert result["error"] == "openai_authentication"
        assert "OPENAI_API_KEY" in result["text"]
    
    def test_user_session_history(self):
        """Test user session history management."""
        session = UserSession()
        user_id = 12345
        
        # Add messages
        session.add_message(user_id, "user", "Hello")
        session.add_message(user_id, "assistant", "Hi there!")
        
        # Get history
        history = session.get_history(user_id)
        
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
    
    def test_user_session_clear(self):
        """Test clearing user session history."""
        session = UserSession()
        user_id = 12345
        
        # Add messages
        session.add_message(user_id, "user", "Test message")
        assert len(session.get_history(user_id)) == 1
        
        # Clear history
        session.clear_history(user_id)
        assert len(session.get_history(user_id)) == 0
    
    def test_user_mode_management(self):
        """Test user mode setting and retrieval."""
        session = UserSession()
        user_id = 12345
        
        # Default mode
        assert session.get_mode(user_id) == "text"
        
        # Set new mode
        session.set_mode(user_id, "rag")
        assert session.get_mode(user_id) == "rag"
    
    def test_user_voice_management(self):
        """Test user voice setting and retrieval."""
        session = UserSession()
        user_id = 12345
        
        # Default voice
        from config import DEFAULT_VOICE
        assert session.get_voice(user_id) == DEFAULT_VOICE
        
        # Set new voice
        session.set_voice(user_id, "nova")
        assert session.get_voice(user_id) == "nova"


class TestTextHelpers:
    """Test suite for text helper functions."""
    
    def test_truncate_text(self):
        """Test text truncation."""
        from utils.helpers import truncate_text
        
        # Short text
        short = "Hello"
        assert truncate_text(short, 10) == "Hello"
        
        # Long text
        long = "A" * 100
        truncated = truncate_text(long, 10)
        assert len(truncated) == 10
        assert truncated.endswith("...")
    
    def test_format_file_size(self):
        """Test file size formatting."""
        from utils.helpers import format_file_size
        
        assert "B" in format_file_size(500)
        assert "KB" in format_file_size(1024)
        assert "MB" in format_file_size(1024 * 1024)
        assert "GB" in format_file_size(1024 * 1024 * 1024)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


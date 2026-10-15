"""
OpenAI Client for the Personal Assistant Bot.
Provides methods for text generation, vision, STT, and TTS.
"""

from typing import List, Dict, Optional
from openai import AsyncOpenAI, AuthenticationError
from pathlib import Path
import json

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    GPT_MODEL,
    WHISPER_MODEL,
    TTS_MODEL,
    VISION_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)
from utils.logging import logger


class OpenAIClient:
    """Async client for OpenAI API operations."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        logger.info(f"OpenAI client initialized with direct API: {OPENAI_BASE_URL}")

    async def generate_reply_or_image(self, messages: List[Dict], classification_only: bool = False) -> Dict:
        """Let the model answer or request one image using the full conversation."""
        instructions = """Ты — личный ассистент в Telegram с инструментом generate_image.
Если пользователь хочет получить готовое изображение (рисунок, иллюстрацию, обложку,
афишу, открытку, логотип и т.п.), вызови generate_image. Понимай смысл запроса,
учитывай историю, ссылки «это», «его», «к этому посту» и предоставленные материалы.
Передай улучшенный самостоятельный промпт на АНГЛИЙСКОМ языке: предмет,
композицию, стиль, освещение, нужные надписи и факты из контекста. Само описание
пиши по-английски; точные надписи для картинки оставляй в кавычках на языке
пользователя. Этот промпт будет показан пользователю и передан генератору без перевода.
Названия и имена собственные сохраняй точно в исходном написании в кавычках:
например, мастерская «Роса» — это "Роса", а не Ross. Не добавляй надписи,
подписи, логотипы или слоганы, которых пользователь не просил. Если текст на
картинке не запрошен, явно укажи в английском промпте: No text or lettering.
Не выдумывай названия, даты и цены из базы знаний.
Не вызывай инструмент для вопросов о рисовании, просьб написать промпт или код,
отрицаний («не рисуй») и простого обсуждения изображений. Если неясно, что нужно
изобразить, задай короткий уточняющий вопрос. Не обещай прислать картинку текстом:
для создания изображения используй инструмент. В остальных случаях отвечай
пользователю на его языке. За один запрос создавай максимум одно изображение."""
        if classification_only:
            instructions += "\nСейчас только выбери действие. Если изображение не нужно, ответь словом text."
        response = await self.client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": instructions}] + list(messages),
            tools=[{
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Создать и отправить пользователю готовое изображение.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"prompt": {"type": "string", "description": "Improved, self-contained image generation prompt in English. Preserve requested on-image text in its original language, in quotes."}},
                        "required": ["prompt"],
                        "additionalProperties": False,
                    },
                },
            }],
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=MAX_TOKENS,
        )
        message = response.choices[0].message
        calls = message.tool_calls or []
        if calls:
            if len(calls) != 1 or calls[0].function.name != "generate_image":
                raise ValueError("Unexpected image routing tool call")
            arguments = json.loads(calls[0].function.arguments)
            prompt = arguments.get("prompt") if isinstance(arguments, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Image routing returned an empty prompt")
            return {"image_prompt": prompt.strip()}
        text = message.content or message.refusal
        if not isinstance(text, str) or not text.strip():
            raise ValueError("OpenAI returned neither a reply nor an image request")
        return {"text": text.strip()}

    async def improve_image_prompt(self, prompt: str) -> str:
        """Prepare an English prompt for explicit /image requests."""
        improved = await self.generate_text_response([
            {"role": "system", "content": (
                "Rewrite the user's image request as an improved, self-contained image "
                "generation prompt in English. Include subject, composition, visual style "
                "and lighting while preserving the user's intent and factual details. "
                "Preserve proper nouns verbatim in their original spelling; do not translate "
                "or replace them with similar names. Do not invent captions, labels, logos "
                "or slogans. If the user did not request text on the image, include the "
                "instruction 'No text or lettering.' "
                "Keep exact text requested on the image in its original language, in quotes. "
                "Return only the English prompt, without headings, commentary or code fences."
            )},
            {"role": "user", "content": prompt},
        ])
        if not isinstance(improved, str) or not improved.strip():
            raise ValueError("OpenAI returned an empty improved image prompt")
        return improved.strip()
    
    async def generate_text_response(
        self,
        messages: List[Dict[str, str]],
        model: str = GPT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ) -> str:
        """
        Generate text response using GPT model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use
            temperature: Response randomness (0-2)
            max_tokens: Maximum tokens in response
        
        Returns:
            Generated text response
        """
        try:
            logger.debug(f"Generating text response with {model}")
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            logger.info(f"Generated response: {len(result)} characters")
            return result
            
        except AuthenticationError:
            logger.error("OpenAI rejected the API key from .env")
            raise
        except Exception as e:
            logger.error(f"Error generating text response: {e}")
            raise
    
    async def analyze_image(
        self,
        image_url: str,
        prompt: str = "Опиши это изображение подробно. Что ты видишь?",
        model: str = VISION_MODEL
    ) -> str:
        """
        Analyze an image using GPT-4 Vision.
        
        Args:
            image_url: URL or base64 encoded image
            prompt: Analysis prompt
            model: Vision model to use
        
        Returns:
            Image analysis result
        """
        try:
            logger.debug(f"Analyzing image with {model}")
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=MAX_TOKENS
            )
            
            result = response.choices[0].message.content
            logger.info(f"Image analyzed: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_file_path: Path,
        model: str = WHISPER_MODEL
    ) -> str:
        """
        Transcribe audio file to text using Whisper.
        
        Args:
            audio_file_path: Path to audio file
            model: Whisper model to use
        
        Returns:
            Transcribed text
        """
        try:
            logger.debug(f"Transcribing audio: {audio_file_path}")
            
            with open(audio_file_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text"
                )
            
            logger.info(f"Audio transcribed: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = TTS_MODEL,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate speech from text using TTS.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, nova, fable, onyx, shimmer)
            model: TTS model to use
            output_path: Path to save audio file
        
        Returns:
            Path to generated audio file
        """
        try:
            logger.debug(f"Generating speech with voice: {voice}")
            
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Default output path
            if output_path is None:
                from config import DATA_DIR
                import uuid
                output_path = DATA_DIR / f"tts_{uuid.uuid4()}.mp3"
            
            # Save audio to file
            response.stream_to_file(str(output_path))
            
            logger.info(f"Speech generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            raise


# Global client instance
openai_client = OpenAIClient()


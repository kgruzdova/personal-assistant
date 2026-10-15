"""
Request Router.
Routes different types of requests to appropriate handlers.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from openai import AuthenticationError

from services.openai_client import openai_client
from services.stt import transcribe_voice_message
from services.tts import generate_voice_response
from services.vision import analyze_image
from services.image_generation import generate_image
from utils.logging import logger
from utils.helpers import user_sessions
from config import BotMode, IMAGE_DEFAULT_SIZE, IMAGE_DEFAULT_QUALITY


async def route_text_request(
    user_id: int,
    text: str,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route text request to appropriate handler.
    
    Args:
        user_id: User ID
        text: User's text message
        mode: Bot mode (text, rag, etc.)
    
    Returns:
        Response dictionary with 'text' and optional 'voice_path' and 'image_path'
    """
    try:
        # Determine mode
        if mode is None:
            mode = user_sessions.get_mode(user_id)
        
        # Get conversation history
        history = list(user_sessions.get_history(user_id))

        # The same model call answers text or selects image generation.
        if mode == BotMode.RAG:
            from rag.query import query_knowledge_base
            decision = await query_knowledge_base(text, history, allow_image_generation=True)
        else:
            decision = await openai_client.generate_reply_or_image(
                history + [{"role": "user", "content": text}]
            )

        if decision.get("image_prompt"):
            logger.info(f"Image generation request detected for user {user_id}")
            return await route_image_generation_request(
                user_id=user_id,
                prompt=decision["image_prompt"],
                original_text=text,
                prompt_is_improved=True,
            )
        
        # Add user message to history
        user_sessions.add_message(user_id, "user", text)
        
        response_text = decision["text"]
        
        # Add assistant response to history
        user_sessions.add_message(user_id, "assistant", response_text)
        
        logger.info(f"Text request processed for user {user_id}")
        return {
            "text": response_text,
            "mode": mode
        }
        
    except AuthenticationError:
        logger.error("OpenAI rejected the API key from .env")
        return {
            "text": "OpenAI не принял API-ключ. Проверьте OPENAI_API_KEY в файле .env.",
            "error": "openai_authentication",
        }
    except Exception as e:
        logger.error(f"Error routing text request: {e}")
        return {
            "text": "Извините, произошла ошибка при обработке запроса.",
            "error": str(e)
        }


async def route_voice_request(
    user_id: int,
    voice_path: Path
) -> Dict[str, Any]:
    """
    Route voice request: transcribe, process, and generate voice response.
    
    Args:
        user_id: User ID
        voice_path: Path to voice message file
    
    Returns:
        Response dictionary with 'text', 'transcription', 'voice_path', and optional 'image_path'
    """
    try:
        # Transcribe voice to text
        logger.debug(f"Transcribing voice for user {user_id}")
        transcription = await transcribe_voice_message(voice_path)
        
        # Process text request (may include image generation)
        text_response = await route_text_request(user_id, transcription)
        
        # Check if response contains an image
        if text_response.get('has_image'):
            # If image was generated, return without voice response
            logger.info(f"Voice request with image generation for user {user_id}")
            return {
                "text": text_response["text"],
                "transcription": transcription,
                "has_image": True,
                "image_path": text_response.get("image_path"),
                "revised_prompt": text_response.get("revised_prompt"),
                "generation_prompt": text_response["generation_prompt"],
                "voice_path": None  # No voice response when image is generated
            }
        
        # Generate voice response for normal text
        user_voice = user_sessions.get_voice(user_id)
        logger.debug(f"Generating voice response with voice: {user_voice}")
        voice_response_path = await generate_voice_response(
            text_response["text"],
            voice=user_voice
        )
        
        logger.info(f"Voice request processed for user {user_id}")
        return {
            "text": text_response["text"],
            "transcription": transcription,
            "voice_path": voice_response_path,
            "has_image": False
        }
        
    except Exception as e:
        logger.error(f"Error routing voice request: {e}")
        return {
            "text": "Извините, произошла ошибка при обработке голосового сообщения.",
            "error": str(e)
        }


async def route_image_request(
    user_id: int,
    image_path: Optional[Path] = None,
    image_url: Optional[str] = None,
    caption: Optional[str] = None
) -> Dict[str, Any]:
    """
    Route image request: analyze image with Vision API.
    
    Args:
        user_id: User ID
        image_path: Local path to image
        image_url: URL to image
        caption: Optional caption/question about image
    
    Returns:
        Response dictionary with 'text' (analysis result)
    """
    try:
        # Prepare custom prompt if caption provided
        custom_prompt = None
        if caption:
            custom_prompt = f"{caption}\n\nПроанализируй изображение с учетом этого вопроса."
        
        # Analyze image
        logger.debug(f"Analyzing image for user {user_id}")
        analysis = await analyze_image(
            image_path=image_path,
            image_url=image_url,
            custom_prompt=custom_prompt
        )
        
        # Add to conversation history
        context = f"[Пользователь отправил изображение]"
        if caption:
            context += f" с подписью: {caption}"
        user_sessions.add_message(user_id, "user", context)
        user_sessions.add_message(user_id, "assistant", analysis)
        
        logger.info(f"Image request processed for user {user_id}")
        return {
            "text": analysis
        }
        
    except Exception as e:
        logger.error(f"Error routing image request: {e}")
        return {
            "text": "Извините, произошла ошибка при анализе изображения.",
            "error": str(e)
        }


async def route_rag_request(
    user_id: int,
    query: str
) -> Dict[str, Any]:
    """
    Route RAG request: query knowledge base.
    
    Args:
        user_id: User ID
        query: User's query
    
    Returns:
        Response dictionary with 'text' and 'sources'
    """
    try:
        from rag.query import query_knowledge_base
        
        # Get conversation history
        history = user_sessions.get_history(user_id)
        
        # Query knowledge base
        logger.debug(f"Querying knowledge base for user {user_id}")
        response = await query_knowledge_base(query, history)
        
        # Add to history
        user_sessions.add_message(user_id, "user", query)
        user_sessions.add_message(user_id, "assistant", response)
        
        logger.info(f"RAG request processed for user {user_id}")
        return {
            "text": response,
            "mode": "rag"
        }
        
    except Exception as e:
        logger.error(f"Error routing RAG request: {e}")
        # Fallback to regular text response
        return await route_text_request(user_id, query, mode=BotMode.TEXT)


async def route_image_generation_request(
    user_id: int,
    prompt: str,
    original_text: str,
    size: str = IMAGE_DEFAULT_SIZE,
    quality: str = IMAGE_DEFAULT_QUALITY,
    style: Optional[str] = None,
    prompt_is_improved: bool = False,
) -> Dict[str, Any]:
    """
    Route an image request to the OpenAI image model.
    
    Args:
        user_id: User ID
        prompt: Processed prompt for image generation
        original_text: Original user text
        size: Image size (1024x1024, 1024x1536, 1536x1024)
        quality: Image quality (low, medium, high, auto)
        style: Optional visual style
    
    Returns:
        Response dictionary with 'text', 'image_path', and generation details
    """
    try:
        if not prompt_is_improved:
            prompt = await openai_client.improve_image_prompt(prompt)
        logger.info(f"Generating image for user {user_id}: {prompt[:100]}...")
        
        # Generate image
        result = await generate_image(
            prompt=prompt,
            size=size,
            quality=quality,
            style=style
        )
        
        # Prepare response text
        response_text = "🎨 Изображение готово!"
        
        # Add to conversation history
        user_sessions.add_message(user_id, "user", f"[Запрос на генерацию изображения: {original_text}]")
        user_sessions.add_message(
            user_id, 
            "assistant", 
            f"[Изображение создано: {result['revised_prompt'][:100]}...]"
        )
        
        logger.info(f"Image generation completed for user {user_id}")
        return {
            "text": response_text,
            "image_path": result['image_path'],
            "revised_prompt": result['revised_prompt'],
            "original_prompt": result['original_prompt'],
            "generation_prompt": result["generation_prompt"],
            "has_image": True
        }
        
    except Exception as e:
        logger.error("Image request failed: %s (code=%s)", type(e).__name__, getattr(e, "code", None))
        
        # Add error to history
        user_sessions.add_message(user_id, "user", f"[Запрос на генерацию изображения: {original_text}]")
        
        error_message = "❌ Извините, произошла ошибка при генерации изображения. "
        
        code = str(getattr(e, "code", "") or "").lower()
        if isinstance(e, AuthenticationError):
            error_message += "OpenAI не принял OPENAI_API_KEY из файла .env."
        elif "billing" in code or "quota" in code:
            error_message += "Возможно, исчерпан лимит API. Проверьте баланс OpenAI."
        elif "content_policy" in code or "moderation" in code:
            error_message += "Запрос нарушает политику контента OpenAI."
        else:
            error_message += f"Попробуйте еще раз или перефразируйте запрос."
        
        user_sessions.add_message(user_id, "assistant", error_message)
        
        return {
            "text": error_message,
            "error": str(e),
            "has_image": False
        }


"""
Image Generation Service.
Uses the official OpenAI Images API to generate images from text prompts.
"""

import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4
import base64

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, DATA_DIR,
    IMAGE_MODEL, IMAGE_DEFAULT_SIZE, IMAGE_DEFAULT_QUALITY,
)
from services.openai_client import openai_client
from utils.logging import logger


# Create temp directory for generated images
GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)


async def detect_image_generation_intent(text: str, conversation_history: list = None) -> Dict[str, Any]:
    """
    Ask the model to interpret an image request in its conversation context.
    
    Args:
        text: User's message text
        conversation_history: Recent conversation history for context
    
    Returns:
        Dictionary with 'needs_generation' (bool) and 'prompt' (str) if detected
    """
    if not text.strip():
        return {"needs_generation": False, "confidence": 0.0}
    messages = list(conversation_history or [])
    messages.append({"role": "user", "content": text})
    decision = await openai_client.generate_reply_or_image(messages, classification_only=True)
    if decision.get("image_prompt"):
        return {"needs_generation": True, "prompt": decision["image_prompt"], "confidence": 1.0}
    return {"needs_generation": False, "confidence": 1.0}


async def generate_image(
    prompt: str,
    size: str = IMAGE_DEFAULT_SIZE,
    quality: str = IMAGE_DEFAULT_QUALITY,
    style: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a PNG using GPT Image and save its base64 response locally.
    
    Args:
        prompt: Text description of the image to generate
        size: Image size (1024x1024, 1024x1536, 1536x1024)
        quality: Image quality (low, medium, high, auto)
        style: Optional visual style to include in the prompt
    
    Returns:
        Dictionary with 'image_path', 'revised_prompt', and 'url'
    """
    try:
        logger.info("Generating image with %s", IMAGE_MODEL)
        # Accept legacy callers while sending only GPT Image parameters.
        size = {"1024x1792": "1024x1536", "1792x1024": "1536x1024"}.get(size, size)
        quality = {"standard": "medium", "hd": "high"}.get(quality, quality)
        image_prompt = f"{prompt}\nVisual style: {style}." if style else prompt
        result = await openai_client.client.with_options(timeout=180.0, max_retries=0).images.generate(
            model=IMAGE_MODEL, prompt=image_prompt, n=1,
            size=size, quality=quality, output_format="png",
        )
        if not result.data or not result.data[0].b64_json:
            raise ValueError("OpenAI returned no image data")
        image_data = result.data[0]
        image_bytes = base64.b64decode(image_data.b64_json, validate=True)
        if not image_bytes:
            raise ValueError("OpenAI returned an empty image")
        image_path = GENERATED_IMAGES_DIR / f"generated_{uuid4().hex}.png"
        async with aiofiles.open(image_path, "wb") as image_file:
            await image_file.write(image_bytes)
        revised_prompt = image_data.revised_prompt or prompt
        
        return {
            "image_path": image_path,
            "revised_prompt": revised_prompt,
            "url": image_data.url,
            "original_prompt": prompt,
            "generation_prompt": image_prompt,
        }
        
    except Exception as e:
        logger.error("Image generation failed: %s (code=%s)", type(e).__name__, getattr(e, "code", None))
        raise


async def download_image(url: str) -> Path:
    """
    Download image from URL and save to local file.
    
    Args:
        url: Image URL
    
    Returns:
        Path to downloaded image
    """
    try:
        # Generate unique filename
        filename = f"generated_{uuid4().hex}.png"
        filepath = GENERATED_IMAGES_DIR / filename
        
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image: {response.status}")
                
                # Save to file
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(await response.read())
        
        logger.info(f"Image downloaded to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise


async def generate_image_variations(
    image_path: Path,
    n: int = 1,
    size: str = "1024x1024"
) -> list:
    """
    Generate variations of an existing image.
    
    Args:
        image_path: Path to source image
        n: Number of variations to generate (1-10)
        size: Image size
    
    Returns:
        List of paths to generated variations
    """
    try:
        logger.info(f"Generating {n} variations of image: {image_path}")
        
        # Prepare multipart form data
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        data = aiohttp.FormData()
        data.add_field('image', 
                      open(image_path, 'rb'),
                      filename=image_path.name,
                      content_type='image/png')
        data.add_field('n', str(n))
        data.add_field('size', size)
        
        api_url = f"{OPENAI_BASE_URL}/images/variations"
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers=headers,
                data=data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"DALL-E variations API error: {error_text}")
                    raise Exception(f"API error: {response.status}")
                
                result = await response.json()
        
        # Download all variations
        variation_paths = []
        for i, image_data in enumerate(result['data']):
            url = image_data['url']
            path = await download_image(url)
            variation_paths.append(path)
        
        logger.info(f"Generated {len(variation_paths)} variations successfully")
        return variation_paths
        
    except Exception as e:
        logger.error(f"Error generating variations: {e}")
        raise


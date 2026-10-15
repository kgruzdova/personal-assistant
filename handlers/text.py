"""
Text Message Handler.
Handles regular text messages from users using pyTelegramBotAPI.
"""

from telebot import types
from bot import bot
from services.router import route_text_request
from utils.logging import logger
from utils.helpers import user_sessions
from config import BotMode
from handlers.generated_image import send_generated_image


@bot.message_handler(commands=['mode'])
async def cmd_mode(message: types.Message):
    """Handle /mode command - change bot mode."""
    user_id = message.from_user.id
    
    # Parse command arguments
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        # Show current mode and available modes
        current_mode = user_sessions.get_mode(user_id)
        
        mode_info = f"""🔧 **Текущий режим:** `{current_mode}`

**Доступные режимы:**

• `text` - Текстовый режим (GPT-4o)
• `voice` - Голосовой режим (с TTS ответами)
• `vision` - Анализ изображений (GPT-4 Vision)
• `rag` - Работа с базой знаний

**Генерация изображений:**
Просто напишите "Нарисуй...", "Создай изображение..." или "Сгенерируй картинку..."
ИИ автоматически определит запрос и создаст изображение.

**Использование:**
/mode <название_режима>

**Примеры:**
/mode text
/mode rag"""
        
        await bot.send_message(message.chat.id, mode_info)
        return
    
    # Set new mode
    new_mode = args[1].lower()
    valid_modes = [BotMode.TEXT, BotMode.VOICE, BotMode.VISION, BotMode.RAG]
    
    if new_mode not in valid_modes:
        await bot.send_message(
            message.chat.id,
            f"❌ Неизвестный режим: `{new_mode}`\n\n"
            f"Доступные режимы: {', '.join(valid_modes)}"
        )
        return
    
    user_sessions.set_mode(user_id, new_mode)
    logger.info(f"User {user_id} switched to mode: {new_mode}")
    
    mode_descriptions = {
        BotMode.TEXT: "📝 Текстовый режим - обычный диалог с GPT-4o",
        BotMode.VOICE: "🎤 Голосовой режим - ответы будут приходить голосом",
        BotMode.VISION: "📸 Режим Vision - отправляйте изображения для анализа",
        BotMode.RAG: "📚 Режим RAG - работа с базой знаний"
    }
    
    await bot.send_message(
        message.chat.id,
        f"✅ Режим изменен!\n\n{mode_descriptions[new_mode]}"
    )


@bot.message_handler(commands=['image'])
async def cmd_image(message: types.Message):
    """Handle /image command - generate image with specific parameters."""
    user_id = message.from_user.id
    
    # Parse command arguments
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        help_text = """🎨 **Генерация изображений**

**Автоматическая генерация:**
Просто напишите "Нарисуй...", "Создай изображение..." и ИИ автоматически создаст картинку.

**Примеры:**
• Нарисуй кота в космосе
• Создай изображение футуристического города
• Сгенерируй картинку заката на море

**Прямая команда:**
/image <описание>

Бот создаёт изображения через OpenAI и отправляет готовую картинку в этот чат."""
        
        await bot.send_message(message.chat.id, help_text)
        return
    
    prompt = args[1]
    
    logger.info(f"Direct image generation request from user {user_id}")
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Generate image directly
        from services.router import route_image_generation_request
        
        response = await route_image_generation_request(
            user_id=user_id,
            prompt=prompt,
            original_text=prompt
        )
        
        if response.get('has_image') and response.get('image_path'):
            await send_generated_image(message.chat.id, response)
        else:
            await bot.send_message(message.chat.id, response["text"])
    
    except Exception as e:
        logger.error(f"Error in /image command: {e}", exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при генерации изображения.\n"
            "Попробуйте еще раз или перефразируйте запрос."
        )


@bot.message_handler(func=lambda message: message.content_type == 'text' and not message.text.startswith('/'))
async def handle_text_message(message: types.Message):
    """Handle regular text messages."""
    user_id = message.from_user.id
    text = message.text
    
    logger.info(f"Text message from user {user_id}: {text[:50]}...")
    
    # Show typing indicator
    await bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Route request
        response = await route_text_request(user_id, text)
        
        # Check if response contains an image
        if response.get('has_image') and response.get('image_path'):
            await send_generated_image(message.chat.id, response)
            logger.info(f"Image sent to user {user_id}")
            return
        
        # Check mode for voice response
        mode = user_sessions.get_mode(user_id)
        
        if mode == BotMode.VOICE:
            # Generate voice response
            from services.tts import generate_voice_response
            from utils.helpers import cleanup_file
            
            voice_path = await generate_voice_response(
                response["text"],
                voice=user_sessions.get_voice(user_id)
            )
            
            try:
                # Send text first
                await bot.send_message(message.chat.id, response["text"])
                
                # Then send voice
                with open(voice_path, 'rb') as audio:
                    await bot.send_voice(message.chat.id, audio)
                
            finally:
                # Cleanup
                cleanup_file(voice_path)
        else:
            # Send text response
            await bot.send_message(message.chat.id, response["text"])
    
    except Exception as e:
        logger.error(f"Error handling text message: {e}", exc_info=True)
        await bot.send_message(
            message.chat.id,
            "❌ Произошла ошибка при обработке сообщения.\n"
            "Попробуйте еще раз или используйте /reset для сброса."
        )

"""
Start and Help Command Handlers.
Handles /start and /help commands using pyTelegramBotAPI.
"""

from telebot import types
from html import escape
from bot import bot
from utils.logging import logger
from utils.helpers import user_sessions
from config import BotMode, DEFAULT_MODE


@bot.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    """Handle /start command."""
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    
    logger.info(f"User {user_id} started the bot")
    
    # Initialize user session
    user_sessions.set_mode(user_id, DEFAULT_MODE)
    
    welcome_text = f"""👋 Привет, {user_name}!

Я - твой личный мультимодальный ассистент с поддержкой:

🔤 **Текстовые запросы** - спрашивай что угодно
🎤 **Голосовые сообщения** - отправь голос, получи голосовой ответ
📸 **Анализ изображений** - распознавание объектов и документов
📚 **База знаний (RAG)** - поиск по загруженным документам

**Основные команды:**

/help - список всех команд и возможностей
/mode - смена режима работы
/voice - выбор голоса для ответов
/reset - очистка истории диалога
/stats - статистика базы знаний

**Режимы работы:**

• `text` - обычный текстовый режим (GPT-4)
• `voice` - голосовой режим
• `vision` - анализ изображений
• `rag` - работа с базой знаний

Просто начни общаться! 🚀"""
    
    await bot.send_message(message.chat.id, welcome_text)


@bot.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    """Handle /help command."""
    user_id = message.from_user.id
    logger.info(f"User {user_id} requested help")
    
    help_text = """📖 **Полное руководство по боту**

**🔤 Текстовый режим**
Просто напиши сообщение - я отвечу используя GPT-4o.

**🎤 Голосовой режим**
1. Отправь голосовое сообщение
2. Я распознаю речь через Whisper
3. Обработаю запрос
4. Отвечу голосом + текстом

**📸 Режим Vision**
1. Отправь фото
2. Можешь добавить подпись с вопросом
3. Получи детальный анализ изображения

**📚 Режим RAG (База знаний)**
1. Переключись: /mode rag
2. Загрузи документы в папку data/documents/
3. Задавай вопросы по документам
4. Получай ответы с указанием источников

**⚙️ Команды управления:**

/mode <режим> - переключить режим
  • text - текстовый (по умолчанию)
  • voice - голосовой
  • vision - анализ изображений
  • rag - база знаний

/voice <имя> - выбрать голос
  • alloy - нейтральный (по умолчанию)
  • echo - мужской
  • nova - женский
  • fable - британский
  • onyx - глубокий мужской
  • shimmer - теплый женский

/reset - очистить историю диалога
/stats - статистика базы знаний
/voices - список доступных голосов

**💡 Примеры использования:**

1. "Объясни квантовую физику простыми словами"
2. [Голосовое] "Какая погода в Москве?"
3. [Фото документа] "Извлеки данные из этого чека"
4. [В режиме RAG] "Найди информацию о проекте X"

**🔧 Технологии:**
• GPT-4o для текста
• GPT-4 Vision для изображений
• Whisper для распознавания речи
• TTS-1 для синтеза речи
• ChromaDB + LangChain для RAG

Нужна помощь? Просто спроси! 😊"""
    
    await bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['reset'])
async def cmd_reset(message: types.Message):
    """Handle /reset command - clear conversation history."""
    user_id = message.from_user.id
    
    user_sessions.clear_history(user_id)
    logger.info(f"User {user_id} cleared conversation history")
    
    await bot.send_message(
        message.chat.id,
        "✅ История диалога очищена!\n\n"
        "Начнем с чистого листа. Чем могу помочь?"
    )


@bot.message_handler(commands=['stats'])
async def cmd_stats(message: types.Message):
    """Handle /stats command - show knowledge base statistics."""
    user_id = message.from_user.id
    logger.info(f"User {user_id} requested stats")
    
    try:
        from rag.query import get_knowledge_base_stats
        
        stats = get_knowledge_base_stats()
        
        if "error" in stats:
            await bot.send_message(
                message.chat.id,
                f"⚠️ Ошибка получения статистики:\n{escape(str(stats['error']))}",
                parse_mode="HTML",
            )
            return
        
        total_docs = stats.get("total_documents", 0)
        persist_dir = stats.get("persist_directory", "N/A")
        
        stats_text = f"""📊 <b>Статистика базы знаний</b>

📄 Документов в индексе: {total_docs}
💾 Директория: <code>{escape(str(persist_dir))}</code>

{"✅ База знаний готова к использованию!" if total_docs > 0 else "⚠️ База знаний пуста. Добавьте документы в data/documents/"}

Используйте /mode rag для работы с базой знаний."""
        
        await bot.send_message(message.chat.id, stats_text, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        await bot.send_message(
            message.chat.id,
            "⚠️ Ошибка получения статистики базы знаний.",
            parse_mode="HTML",
        )

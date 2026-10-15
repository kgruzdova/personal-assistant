"""
Простой тест бота - проверка что всё работает.
"""

import asyncio
from bot import bot
from utils.logging import logger


async def test():
    """Тестовая функция."""
    try:
        logger.info("Testing bot connection...")
        bot_info = await bot.get_me()
        logger.info(f"✅ Bot is working! @{bot_info.username}")
        logger.info(f"Bot ID: {bot_info.id}")
        logger.info(f"Bot name: {bot_info.first_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return False
    finally:
        await bot.close_session()


if __name__ == "__main__":
    result = asyncio.run(test())
    if result:
        print("\n✅ Бот настроен правильно! Можете запускать: python main.py")
    else:
        print("\n❌ Проверьте TELEGRAM_BOT_TOKEN в .env файле")


"""
Main Entry Point.
Starts the Telegram bot using pyTelegramBotAPI.
"""

import asyncio
import sys

from bot import bot
from utils.logging import logger


async def setup_bot():
    """Setup bot with handlers and initialize RAG if needed."""
    logger.info("Bot starting up...")
    
    # Import handlers (they will register themselves via decorators)
    try:
        from handlers import start, text, voice, image, document_upload
        logger.info("Handlers imported successfully")
    except Exception as e:
        logger.error(f"Error importing handlers: {e}", exc_info=True)
        raise
    
    # Initialize RAG index if documents exist
    try:
        from rag.index import vector_index
        from config import DOCUMENTS_DIR
        
        # Check if documents directory has files
        docs = [d for d in DOCUMENTS_DIR.rglob('*')
                if d.is_file() and d.suffix.lower() in ['.pdf', '.txt', '.md']]
        
        if docs:
            logger.info(f"Found {len(docs)} documents, checking for changes...")
            count = vector_index.index_documents_directory(force_reindex=False)
            logger.info(f"Indexed {count} new or updated document chunks")
        else:
            logger.info("No documents found in data/documents/")
    
    except Exception as e:
        logger.warning(f"Could not initialize RAG index: {e}")
    
    try:
        bot_info = await bot.get_me()
        logger.info(f"Bot started: @{bot_info.username}")
    except Exception as e:
        logger.error(f"Could not get bot info: {e}")


async def shutdown_bot():
    """Actions to perform on bot shutdown."""
    logger.info("Bot shutting down...")
    try:
        await bot.close_session()
    except:
        pass
    logger.info("Bot shutdown complete")


async def main():
    """Main function to run the bot."""
    try:
        # Setup bot
        await setup_bot()
        
        # Start polling
        logger.info("Starting bot polling...")
        await bot.infinity_polling(
            timeout=10,
            skip_pending=True
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await shutdown_bot()


if __name__ == "__main__":
    try:
        logger.info("="*60)
        logger.info("Personal Assistant Bot - Starting")
        logger.info("="*60)
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)

"""Send the exact generation prompt before its picture and release the file."""

from html import escape

from bot import bot
from utils.helpers import cleanup_file


async def send_generated_image(chat_id: int, response: dict):
    image_path = response["image_path"]
    try:
        prompt = response["generation_prompt"]
        # Keep the full prompt, including long ones, within Telegram message limits.
        # 1800 code points also leave space when characters take two UTF-16 units.
        for offset in range(0, len(prompt), 1800):
            heading = "Улучшенный промпт на английском языке:" if offset == 0 else "Продолжение промпта:"
            await bot.send_message(
                chat_id,
                f"<b>{heading}</b>\n<pre>{escape(prompt[offset:offset + 1800])}</pre>",
                parse_mode="HTML",
            )
        await bot.send_chat_action(chat_id, "upload_photo")
        with open(image_path, "rb") as photo:
            await bot.send_photo(
                chat_id,
                photo,
                caption=escape(response.get("text", "🎨 Изображение готово!")[:900]),
                parse_mode="HTML",
            )
    finally:
        cleanup_file(image_path)

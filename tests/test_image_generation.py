"""Image routing, OpenAI payloads and Telegram delivery without live API calls."""

import base64
import json
from html import escape, unescape
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from services.openai_client import openai_client
from services.image_generation import detect_image_generation_intent, generate_image
from utils.helpers import user_sessions


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)


def completion(text=None, prompt=None, arguments=None):
    calls = []
    if prompt is not None or arguments is not None:
        calls = [SimpleNamespace(function=SimpleNamespace(
            name="generate_image",
            arguments=arguments if arguments is not None else json.dumps({"prompt": prompt}),
        ))]
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=text, refusal=None, tool_calls=calls,
    ))])


@pytest.fixture
def chat_api(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr(openai_client.client.chat.completions, "create", create)
    return create


@pytest.fixture
def image_api(monkeypatch, tmp_path):
    from services import image_generation

    monkeypatch.setattr(image_generation, "GENERATED_IMAGES_DIR", tmp_path)
    generate = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(
        b64_json=base64.b64encode(PNG_BYTES).decode(), revised_prompt=None, url=None,
    )]))
    options = Mock(return_value=SimpleNamespace(images=SimpleNamespace(generate=generate)))
    monkeypatch.setattr(openai_client.client, "with_options", options)
    return generate


@pytest.fixture
def user_id():
    value = 778899
    user_sessions.clear_history(value)
    yield value
    user_sessions.clear_history(value)


@pytest.mark.asyncio
async def test_regular_message_uses_one_model_call(chat_api, image_api, user_id):
    from services.router import route_text_request

    chat_api.return_value = completion(text="Я могу отвечать на вопросы и создавать изображения.")
    result = await route_text_request(user_id, "Привет, что ты умеешь?", mode="text")

    assert result["text"].startswith("Я могу")
    chat_api.assert_awaited_once()
    image_api.assert_not_awaited()
    assert chat_api.call_args.kwargs["tool_choice"] == "auto"
    assert chat_api.call_args.kwargs["parallel_tool_calls"] is False


@pytest.mark.asyncio
async def test_indirect_request_calls_image_tool_with_history(chat_api, image_api, user_id):
    from services.router import route_text_request

    user_sessions.add_message(user_id, "user", "Мастерская Роса проводит занятия акварелью.")
    user_sessions.add_message(user_id, "assistant", "Подготовим пост о занятиях.")
    prompt = "Акварельная обложка для мастерской Роса с кистями и надписью «Роса»."
    chat_api.return_value = completion(prompt=prompt)
    result = await route_text_request(user_id, "Нужна красивая обложка к этому посту", mode="text")

    assert result["has_image"] is True
    assert result["image_path"].read_bytes() == PNG_BYTES
    assert image_api.call_args.kwargs["prompt"] == prompt
    sent_messages = chat_api.call_args.kwargs["messages"]
    assert sent_messages[1]["content"].startswith("Мастерская Роса")
    assert sent_messages[-1]["content"] == "Нужна красивая обложка к этому посту"
    chat_api.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Как нарисовать кота?", "Не рисуй, напиши промпт для картинки"])
async def test_model_can_decline_generation_despite_keywords(chat_api, image_api, user_id, text):
    from services.router import route_text_request

    chat_api.return_value = completion(text="Вот текстовый ответ.")
    result = await route_text_request(user_id, text, mode="text")
    assert result["text"] == "Вот текстовый ответ."
    image_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_detector_passes_context_to_ai(chat_api):
    history = [{"role": "user", "content": "Наш персонаж — рыжий кот в скафандре."}]
    chat_api.return_value = completion(prompt="Рыжий кот в скафандре на Луне.")
    result = await detect_image_generation_intent("А теперь покажи его на Луне", history)
    assert result["needs_generation"] is True
    assert "скафандре" in result["prompt"]
    assert chat_api.call_args.kwargs["messages"][1] == history[0]
    assert len(history) == 1


@pytest.mark.asyncio
async def test_empty_detection_does_not_call_api(chat_api):
    assert (await detect_image_generation_intent("  "))["needs_generation"] is False
    chat_api.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ['{"prompt": ""}', '{"prompt": null}', '[]', 'not json'])
async def test_bad_tool_arguments_do_not_generate(chat_api, image_api, user_id, arguments):
    from services.router import route_text_request

    chat_api.return_value = completion(arguments=arguments)
    result = await route_text_request(user_id, "Сделай обложку", mode="text")
    assert "error" in result
    image_api.assert_not_awaited()


@pytest.mark.asyncio
async def test_gpt_image_base64_and_parameters(image_api):
    from config import IMAGE_MODEL

    result = await generate_image("Лес", size="1024x1792", quality="standard")
    assert result["image_path"].read_bytes() == PNG_BYTES
    payload = image_api.call_args.kwargs
    assert payload["model"] == IMAGE_MODEL
    assert payload["size"] == "1024x1536"
    assert payload["quality"] == "medium"
    assert payload["output_format"] == "png"
    assert "style" not in payload
    assert "response_format" not in payload
    assert result["generation_prompt"] == payload["prompt"]


@pytest.mark.asyncio
async def test_generated_files_have_unique_names(image_api):
    first = await generate_image("Первый лес")
    second = await generate_image("Второй лес")
    assert first["image_path"] != second["image_path"]


@pytest.mark.asyncio
async def test_empty_image_response_fails(image_api):
    image_api.return_value = SimpleNamespace(data=[])
    with pytest.raises(ValueError, match="no image data"):
        await generate_image("Лес")


@pytest.mark.asyncio
async def test_rag_supplies_document_facts_to_image_planner(chat_api, image_api, user_id, monkeypatch):
    from rag import query
    from services.router import route_text_request

    document = SimpleNamespace(
        page_content="Мастерская Роса: занятие акварелью 20 октября в 18:00.",
        metadata={"source": "sample_workshop.txt"},
    )
    monkeypatch.setattr(query.vector_index, "similarity_search_with_score", Mock(return_value=[(document, 0.1)]))
    chat_api.return_value = completion(prompt="Афиша Роса: акварель, 20 октября, 18:00.")
    result = await route_text_request(user_id, "Сделай афишу мастерской Роса", mode="rag")
    assert result["has_image"] is True
    messages = chat_api.call_args.kwargs["messages"]
    assert any("20 октября в 18:00" in m["content"] for m in messages)
    assert "20 октября" in image_api.call_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_text_handler_delivers_generated_photo(chat_api, image_api, user_id, monkeypatch):
    from handlers import text

    prompt = 'A cinematic cover: a cat in space, with the text "Привет" and <stars> & galaxies.'
    chat_api.return_value = completion(prompt=prompt)
    photo_data = []
    events = []

    async def receive_message(chat_id, message_text, **kwargs):
        events.append("prompt")
        assert "Улучшенный промпт на английском языке:" in message_text
        assert escape(prompt) in message_text
        assert kwargs["parse_mode"] == "HTML"

    async def receive_photo(chat_id, photo, **kwargs):
        events.append("photo")
        photo_data.append((chat_id, photo.read(), photo.name, kwargs))

    monkeypatch.setattr(text.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(text.bot, "send_photo", AsyncMock(side_effect=receive_photo))
    monkeypatch.setattr(text.bot, "send_message", AsyncMock(side_effect=receive_message))
    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=123), text="Сделай обложку")
    await text.handle_text_message(message)

    assert photo_data[0][0] == 123
    assert photo_data[0][1] == PNG_BYTES
    assert photo_data[0][3]["parse_mode"] == "HTML"
    from pathlib import Path
    assert not Path(photo_data[0][2]).exists()
    assert image_api.call_args.kwargs["prompt"] == prompt
    assert events == ["prompt", "photo"]


@pytest.mark.asyncio
async def test_voice_handler_sends_the_used_english_prompt_first(chat_api, image_api, user_id, monkeypatch, tmp_path):
    from handlers import voice
    from services import router

    prompt = "A watercolor painting of a ginger cat on the Moon, soft blue lighting."
    chat_api.return_value = completion(prompt=prompt)
    voice_file = tmp_path / "voice.ogg"
    voice_file.write_bytes(b"voice")
    monkeypatch.setattr(voice.bot, "get_file", AsyncMock(return_value=SimpleNamespace(file_path="voice.ogg")))
    monkeypatch.setattr(voice.bot, "download_file", AsyncMock(return_value=b"voice"))
    monkeypatch.setattr(voice, "save_file_async", AsyncMock(return_value=voice_file))
    monkeypatch.setattr(router, "transcribe_voice_message", AsyncMock(return_value="Нарисуй кота на Луне"))
    tts = AsyncMock()
    monkeypatch.setattr(router, "generate_voice_response", tts)
    events = []

    async def receive_message(chat_id, message_text, **kwargs):
        if "Улучшенный промпт" in message_text:
            assert escape(prompt) in message_text
            events.append("prompt")
        else:
            events.append("transcription")

    async def receive_photo(chat_id, photo, **kwargs):
        assert photo.read() == PNG_BYTES
        events.append("photo")

    monkeypatch.setattr(voice.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(voice.bot, "send_message", AsyncMock(side_effect=receive_message))
    monkeypatch.setattr(voice.bot, "send_photo", AsyncMock(side_effect=receive_photo))
    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=123), voice=SimpleNamespace(file_id="voice"))
    await voice.handle_voice_message(message)

    assert events == ["transcription", "prompt", "photo"]
    assert image_api.call_args.kwargs["prompt"] == prompt
    tts.assert_not_awaited()
    assert not voice_file.exists()


@pytest.mark.asyncio
async def test_explicit_image_command_improves_prompt(image_api, user_id, monkeypatch):
    from handlers import text

    prompt = "A luminous watercolor landscape of snow-capped mountains at sunrise."
    improve = AsyncMock(return_value=prompt)
    monkeypatch.setattr(openai_client, "improve_image_prompt", improve)
    monkeypatch.setattr(text.bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(text.bot, "send_message", AsyncMock())
    monkeypatch.setattr(text.bot, "send_photo", AsyncMock())
    message = SimpleNamespace(from_user=SimpleNamespace(id=user_id), chat=SimpleNamespace(id=123), text="/image горы на рассвете")
    await text.cmd_image(message)

    improve.assert_awaited_once_with("горы на рассвете")
    assert image_api.call_args.kwargs["prompt"] == prompt
    assert escape(prompt) in text.bot.send_message.call_args.args[1]
    text.bot.send_photo.assert_awaited_once()


@pytest.mark.asyncio
async def test_long_prompt_is_sent_in_full_before_photo(tmp_path, monkeypatch):
    from handlers.generated_image import bot, send_generated_image

    path = tmp_path / "generated.png"
    path.write_bytes(PNG_BYTES)
    prompt = "A detailed scene with <stars> & planets. " * 150
    chunks = []
    events = []

    async def receive_message(chat_id, text, **kwargs):
        chunks.append(unescape(text.split("<pre>", 1)[1].split("</pre>", 1)[0]))
        events.append("prompt")

    async def receive_photo(*args, **kwargs):
        events.append("photo")

    monkeypatch.setattr(bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(bot, "send_message", AsyncMock(side_effect=receive_message))
    monkeypatch.setattr(bot, "send_photo", AsyncMock(side_effect=receive_photo))
    await send_generated_image(123, {"image_path": path, "generation_prompt": prompt})

    assert "".join(chunks) == prompt
    assert events == ["prompt"] * len(chunks) + ["photo"]
    assert not path.exists()


@pytest.mark.asyncio
async def test_failed_telegram_upload_still_cleans_file(tmp_path, monkeypatch):
    from handlers.generated_image import bot, send_generated_image

    path = tmp_path / "generated.png"
    path.write_bytes(PNG_BYTES)
    monkeypatch.setattr(bot, "send_chat_action", AsyncMock())
    monkeypatch.setattr(bot, "send_message", AsyncMock())
    monkeypatch.setattr(bot, "send_photo", AsyncMock(side_effect=RuntimeError("upload failed")))
    with pytest.raises(RuntimeError, match="upload failed"):
        await send_generated_image(123, {"image_path": path, "text": "Готово", "generation_prompt": "A forest"})
    assert not path.exists()

"""Regression checks for Telegram formatting in /stats."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from handlers import start


@pytest.mark.asyncio
@pytest.mark.parametrize("stats, expected", [
    (
        {"total_documents": 7, "persist_directory": r"E:\Personal_Assistant\chroma_db & <cache>"},
        r"<code>E:\Personal_Assistant\chroma_db &amp; &lt;cache&gt;</code>",
    ),
    ({"total_documents": 0, "persist_directory": "data/chroma_db"}, "База знаний пуста"),
    ({"error": "bad_index: <invalid> & unavailable"}, "bad_index: &lt;invalid&gt; &amp; unavailable"),
])
async def test_stats_sends_safe_html(monkeypatch, stats, expected):
    # Supply statistics without opening the real RAG database.
    monkeypatch.setitem(sys.modules, "rag.query", SimpleNamespace(get_knowledge_base_stats=Mock(return_value=stats)))
    send = AsyncMock()
    monkeypatch.setattr(start.bot, "send_message", send)
    message = SimpleNamespace(from_user=SimpleNamespace(id=1), chat=SimpleNamespace(id=2))

    await start.cmd_stats(message)

    send.assert_awaited_once()
    assert send.call_args.kwargs["parse_mode"] == "HTML"
    assert expected in send.call_args.args[1]

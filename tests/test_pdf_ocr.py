"""PDF OCR fallback, page provenance and content-based cache without WinRT calls."""

import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from rag.loader import DocumentLoader
from rag import loader as loader_module
from rag import pdf_ocr


@pytest.fixture
def pdf_source(tmp_path, monkeypatch):
    path = tmp_path / "scanned.pdf"
    path.write_bytes(b"test input; PDF parsing is mocked")
    monkeypatch.setattr(pdf_ocr, "OCR_CACHE_DIR", tmp_path / "ocr-cache")
    return path


def test_text_pdf_does_not_use_ocr(pdf_source, monkeypatch):
    native = Document(page_content="Existing text layer.", metadata={"page": 0})
    monkeypatch.setattr(loader_module, "PyPDFLoader", Mock(return_value=Mock(load=Mock(return_value=[native]))))
    recognize = Mock()
    monkeypatch.setattr(loader_module, "recognize_pdf_pages", recognize)

    chunks = DocumentLoader().load_document(pdf_source)

    assert chunks[0].page_content == "Existing text layer."
    recognize.assert_not_called()


def test_mixed_pdf_recognizes_only_pages_without_text(pdf_source, monkeypatch):
    pages = [Document(page_content="Native page.", metadata={"page": 0}), Document(page_content="", metadata={"page": 1})]
    monkeypatch.setattr(loader_module, "PyPDFLoader", Mock(return_value=Mock(load=Mock(return_value=pages))))
    recognize = Mock(return_value={1: {"text": "Распознанная страница.", "language": "ru"}})
    monkeypatch.setattr(loader_module, "recognize_pdf_pages", recognize)

    chunks = DocumentLoader().load_document(pdf_source)

    recognize.assert_called_once_with(pdf_source, [1])
    assert [chunk.metadata["page"] for chunk in chunks] == [0, 1]
    assert chunks[1].metadata["extraction_method"] == "windows_ocr"
    assert chunks[1].metadata["source"] == pdf_source.name
    assert chunks[1].page_content == "Распознанная страница."


def test_blank_pdf_is_an_error_not_a_success(pdf_source, monkeypatch):
    pages = [Document(page_content="", metadata={"page": 0})]
    monkeypatch.setattr(loader_module, "PyPDFLoader", Mock(return_value=Mock(load=Mock(return_value=pages))))
    monkeypatch.setattr(loader_module, "recognize_pdf_pages", Mock(return_value={0: {"text": ""}}))
    with pytest.raises(pdf_ocr.PdfOcrError, match="no extractable text"):
        DocumentLoader().load_document(pdf_source)


def test_blank_page_is_allowed_alongside_readable_text(pdf_source, monkeypatch):
    pages = [Document(page_content="Readable.", metadata={"page": 0}), Document(page_content="", metadata={"page": 1})]
    monkeypatch.setattr(loader_module, "PyPDFLoader", Mock(return_value=Mock(load=Mock(return_value=pages))))
    monkeypatch.setattr(loader_module, "recognize_pdf_pages", Mock(return_value={1: {"text": ""}}))
    chunks = DocumentLoader().load_document(pdf_source)
    assert len(chunks) == 1
    assert chunks[0].page_content == "Readable."


def fake_backend(path, page_numbers, output_path):
    rows = [{"page": page, "text": f"Распознанный текст страницы {page + 1}.", "language": "ru"} for page in page_numbers]
    output_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def test_ocr_cache_is_reused_until_contents_change(pdf_source, monkeypatch):
    backend = Mock(side_effect=fake_backend)
    monkeypatch.setattr(pdf_ocr, "_run_windows_ocr", backend)
    first = pdf_ocr.recognize_pdf_pages(pdf_source, [0, 1])
    second = pdf_ocr.recognize_pdf_pages(pdf_source, [0, 1])
    assert first == second
    assert backend.call_count == 1

    pdf_source.write_bytes(b"changed input")
    pdf_ocr.recognize_pdf_pages(pdf_source, [0, 1])
    assert backend.call_count == 2


def test_partial_cache_does_not_omit_requested_pages(pdf_source, monkeypatch):
    backend = Mock(side_effect=fake_backend)
    monkeypatch.setattr(pdf_ocr, "_run_windows_ocr", backend)
    pdf_ocr.recognize_pdf_pages(pdf_source, [0])
    pages = pdf_ocr.recognize_pdf_pages(pdf_source, [0, 1])
    assert set(pages) == {0, 1}
    assert backend.call_count == 2


def test_ocr_failure_is_retryable(pdf_source, monkeypatch):
    backend = Mock(side_effect=pdf_ocr.PdfOcrError("Unavailable language"))
    monkeypatch.setattr(pdf_ocr, "_run_windows_ocr", backend)
    with pytest.raises(pdf_ocr.PdfOcrError, match="Unavailable language"):
        pdf_ocr.recognize_pdf_pages(pdf_source, [0])
    assert not list(pdf_ocr.OCR_CACHE_DIR.glob("*.json"))
    backend.side_effect = fake_backend
    assert pdf_ocr.recognize_pdf_pages(pdf_source, [0])[0]["text"]


def test_ocr_cannot_silently_omit_a_page(pdf_source, monkeypatch):
    def incomplete(path, pages, output):
        output.write_text('[{"page":0,"text":"First page"}]', encoding="utf-8")
    monkeypatch.setattr(pdf_ocr, "_run_windows_ocr", incomplete)
    with pytest.raises(pdf_ocr.PdfOcrError, match="all requested pages"):
        pdf_ocr.recognize_pdf_pages(pdf_source, [0, 1])
    assert not list(pdf_ocr.OCR_CACHE_DIR.glob("*.json"))


def test_empty_ocr_result_is_not_cached(pdf_source, monkeypatch):
    def empty(path, pages, output):
        output.write_text('[{"page":0,"text":""}]', encoding="utf-8")
    monkeypatch.setattr(pdf_ocr, "_run_windows_ocr", empty)
    assert pdf_ocr.recognize_pdf_pages(pdf_source, [0])[0]["text"] == ""
    assert not list(pdf_ocr.OCR_CACHE_DIR.glob("*.json"))

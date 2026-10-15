"""Incremental indexing against real temporary Chroma databases, without OpenAI."""

import hashlib
import json
import os
from unittest.mock import Mock

import pytest
from langchain_core.embeddings import Embeddings

import rag.index as index_module
from rag.loader import document_loader


class LocalEmbeddings(Embeddings):
    model = "local-test"
    dimensions = 3

    def __init__(self):
        self.batches = []
        self.reject = None

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        if self.reject and any(self.reject in text for text in texts):
            raise RuntimeError("Simulated embedding failure")
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 1.0, 0.0]


@pytest.fixture
def indexing(tmp_path, monkeypatch):
    embeddings = LocalEmbeddings()
    monkeypatch.setattr(index_module, "OpenAIEmbeddings", lambda **kwargs: embeddings)
    docs = tmp_path / "documents"
    docs.mkdir()
    storage = tmp_path / "chroma"
    index = index_module.VectorIndex(storage)
    return index, embeddings, docs, storage


def stored_ids(index, file_path):
    return set(index.vectorstore.get(where={"file_path": str(file_path.resolve())})["ids"])


def record(index, file_path):
    manifest = json.loads(index.manifest_path.read_text(encoding="utf-8"))
    return manifest["files"][index._file_key(file_path)]


def test_restart_skips_unchanged_file_even_if_mtime_changes(indexing, monkeypatch):
    index, embeddings, docs, storage = indexing
    file_path = docs / "Lessons PEm09.txt"
    file_path.write_text("A lesson about memory and indexing.", encoding="utf-8")
    assert index.index_documents_directory(docs) == 1
    first_ids = stored_ids(index, file_path)
    assert record(index, file_path)["sha256"] == hashlib.sha256(file_path.read_bytes()).hexdigest()

    previous_time = file_path.stat().st_mtime
    os.utime(file_path, (previous_time + 10, previous_time + 10))
    loader = Mock(wraps=document_loader.load_document)
    monkeypatch.setattr(document_loader, "load_document", loader)
    restarted = index_module.VectorIndex(storage)
    assert restarted.index_documents_directory(docs) == 0
    assert restarted.last_indexing_summary["skipped"] == 1
    assert len(embeddings.batches) == 1
    loader.assert_not_called()
    assert stored_ids(restarted, file_path) == first_ids


def test_new_files_do_not_reembed_existing_files(indexing):
    index, embeddings, docs, _ = indexing
    first = docs / "first.txt"
    first.write_text("Original information.", encoding="utf-8")
    index.index_documents_directory(docs)
    first_ids = stored_ids(index, first)
    nested = docs / "nested"
    nested.mkdir()
    second = nested / "SECOND.MD"
    second.write_text("New information.", encoding="utf-8")

    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["new"] == 1
    assert index.last_indexing_summary["skipped"] == 1
    assert embeddings.batches[-1] == ["New information."]
    assert stored_ids(index, first) == first_ids


def test_changed_file_replaces_old_chunks_and_preserves_other_files(indexing):
    index, embeddings, docs, _ = indexing
    first = docs / "first.txt"
    first.write_text("Old version of a very long document. " * 120, encoding="utf-8")
    second = docs / "second.txt"
    second.write_text("Independent document.", encoding="utf-8")
    index.index_documents_directory(docs)
    old_ids = stored_ids(index, first)
    other_ids = stored_ids(index, second)
    assert len(old_ids) > 1

    # Keep the timestamp unchanged: content, not mtime, must determine the update.
    previous_time = first.stat().st_mtime
    first.write_text("A short new version.", encoding="utf-8")
    os.utime(first, (previous_time, previous_time))
    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["updated"] == 1
    assert index.last_indexing_summary["skipped"] == 1
    assert not (stored_ids(index, first) & old_ids)
    assert stored_ids(index, second) == other_ids
    assert index.get_stats()["total_documents"] == 2
    assert embeddings.batches[-1] == ["A short new version."]


def test_same_filename_in_different_directories_has_separate_records(indexing):
    index, _, docs, _ = indexing
    first = docs / "same.txt"
    first.write_text("First folder.", encoding="utf-8")
    (docs / "nested").mkdir()
    second = docs / "nested" / "same.txt"
    second.write_text("Second folder.", encoding="utf-8")
    assert index.index_documents_directory(docs) == 2
    first_ids = stored_ids(index, first)
    second.write_text("Updated second folder.", encoding="utf-8")
    assert index.index_documents_directory(docs) == 1
    assert stored_ids(index, first) == first_ids
    assert index.get_stats()["total_documents"] == 2


def test_embedding_failure_keeps_previous_version_and_other_files_continue(indexing):
    index, embeddings, docs, _ = indexing
    first = docs / "a.txt"
    first.write_text("Working version.", encoding="utf-8")
    index.index_documents_directory(docs)
    old_ids = stored_ids(index, first)
    old_record = record(index, first)

    first.write_text("BLOCKED new version.", encoding="utf-8")
    (docs / "b.txt").write_text("Another valid document.", encoding="utf-8")
    embeddings.reject = "BLOCKED"
    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["failed"] == 1
    assert stored_ids(index, first) == old_ids
    assert record(index, first) == old_record

    embeddings.reject = None
    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["updated"] == 1
    assert index.last_indexing_summary["skipped"] == 1
    assert index.get_stats()["total_documents"] == 2


def test_retry_after_partial_write_does_not_create_duplicates(indexing, monkeypatch):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Working version.", encoding="utf-8")
    index.index_documents_directory(docs)
    old_ids = stored_ids(index, file_path)
    old_record = record(index, file_path)
    file_path.write_text("New version.", encoding="utf-8")
    add_documents = index.vectorstore.add_documents

    def write_then_fail(documents, **kwargs):
        add_documents(documents, **kwargs)
        raise RuntimeError("Simulated crash after storage write")

    monkeypatch.setattr(index.vectorstore, "add_documents", write_then_fail)
    assert index.index_documents_directory(docs) == 0
    assert old_ids.issubset(stored_ids(index, file_path))
    assert record(index, file_path) == old_record

    monkeypatch.setattr(index.vectorstore, "add_documents", add_documents)
    assert index.index_documents_directory(docs) == 1
    assert len(stored_ids(index, file_path)) == 1
    assert not (old_ids & stored_ids(index, file_path))


def test_legacy_duplicates_are_replaced_once(indexing):
    index, embeddings, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("One lesson.", encoding="utf-8")
    chunks = document_loader.load_document(file_path)
    index.add_documents(chunks)
    index.add_documents(chunks)
    assert index.get_stats()["total_documents"] == 2

    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["updated"] == 1
    assert index.last_indexing_summary["removed_chunks"] == 2
    assert index.get_stats()["total_documents"] == 1
    calls = len(embeddings.batches)
    assert index.index_documents_directory(docs) == 0
    assert len(embeddings.batches) == calls


def test_missing_database_chunks_are_repaired_even_if_hash_is_unchanged(indexing):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Long lesson content. " * 150, encoding="utf-8")
    count = index.index_documents_directory(docs)
    ids = stored_ids(index, file_path)
    index.vectorstore.delete(ids=[next(iter(ids))])

    assert index.index_documents_directory(docs) == count
    assert stored_ids(index, file_path) == ids


def test_corrupt_manifest_recovers_without_duplicate_chunks(indexing):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Lesson content.", encoding="utf-8")
    index.index_documents_directory(docs)
    ids = stored_ids(index, file_path)
    index.manifest_path.write_text("broken json", encoding="utf-8")

    assert index.index_documents_directory(docs) == 1
    assert stored_ids(index, file_path) == ids
    assert record(index, file_path)["ids"] == list(ids)


def test_manifest_write_failure_is_retryable(indexing, monkeypatch):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Initial version.", encoding="utf-8")
    index.index_documents_directory(docs)
    old_record = record(index, file_path)
    file_path.write_text("Updated version.", encoding="utf-8")
    writer = index._write_manifest
    monkeypatch.setattr(index, "_write_manifest", Mock(side_effect=OSError("disk error")))
    assert index.index_documents_directory(docs) == 0
    assert record(index, file_path) == old_record

    monkeypatch.setattr(index, "_write_manifest", writer)
    assert index.index_documents_directory(docs) == 1
    assert index.get_stats()["total_documents"] == 1
    assert record(index, file_path)["sha256"] != old_record["sha256"]


def test_empty_file_is_remembered_without_embedding_requests(indexing, monkeypatch):
    index, embeddings, docs, _ = indexing
    file_path = docs / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    assert index.index_documents_directory(docs) == 0
    assert index.last_indexing_summary["new"] == 1
    assert record(index, file_path)["ids"] == []
    loader = Mock(wraps=document_loader.load_document)
    monkeypatch.setattr(document_loader, "load_document", loader)
    assert index.index_documents_directory(docs) == 0
    loader.assert_not_called()
    assert embeddings.batches == []


def test_legacy_pdf_with_zero_chunks_is_retried(indexing, monkeypatch):
    from langchain_core.documents import Document

    index, _, docs, _ = indexing
    path = docs / "scanned.pdf"
    path.write_bytes(b"PDF input is mocked")
    index._write_manifest({index._file_key(path): {
        "sha256": index._content_hash(path), "signature": index._index_signature(path), "ids": [],
    }})
    load = Mock(return_value=[Document(page_content="Text recovered with OCR.", metadata={"page": 0, "source": path.name})])
    monkeypatch.setattr(document_loader, "load_document", load)

    assert index.index_documents_directory(docs) == 1
    assert index.last_indexing_summary["updated"] == 1
    assert len(record(index, path)["ids"]) == 1
    assert index.index_documents_directory(docs) == 0
    load.assert_called_once()


def test_pdf_zero_chunks_does_not_delete_old_index_or_mark_success(indexing, monkeypatch):
    from langchain_core.documents import Document

    index, _, docs, _ = indexing
    path = docs / "scanned.pdf"
    path.write_bytes(b"first PDF version")
    load = Mock(return_value=[Document(page_content="Previously readable text.", metadata={"source": path.name})])
    monkeypatch.setattr(document_loader, "load_document", load)
    assert index.index_documents_directory(docs) == 1
    previous = record(index, path)
    ids = stored_ids(index, path)
    path.write_bytes(b"changed PDF version")
    load.return_value = []

    assert index.index_documents_directory(docs) == 0
    assert index.last_indexing_summary["failed"] == 1
    assert index.last_indexing_summary["updated"] == 0
    assert stored_ids(index, path) == ids
    assert record(index, path) == previous


def test_unreadable_changed_file_keeps_previous_chunks(indexing):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Readable version.", encoding="utf-8")
    index.index_documents_directory(docs)
    ids = stored_ids(index, file_path)
    previous = record(index, file_path)
    file_path.write_bytes(b"\xff\xfe")

    assert index.index_documents_directory(docs) == 0
    assert index.last_indexing_summary["failed"] == 1
    assert stored_ids(index, file_path) == ids
    assert record(index, file_path) == previous


def test_changing_file_during_loading_does_not_commit_stale_hash(indexing, monkeypatch):
    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Initial version.", encoding="utf-8")
    index.index_documents_directory(docs)
    ids = stored_ids(index, file_path)
    file_path.write_text("New version.", encoding="utf-8")
    loader = document_loader.load_document

    def load_and_change(path):
        chunks = loader(path)
        path.write_text("Changed during loading.", encoding="utf-8")
        return chunks

    monkeypatch.setattr(document_loader, "load_document", load_and_change)
    assert index.index_documents_directory(docs) == 0
    assert index.last_indexing_summary["failed"] == 1
    assert stored_ids(index, file_path) == ids


def test_force_reindex_clears_collection_through_chroma_api(indexing):
    index, embeddings, docs, _ = indexing
    (docs / "lesson.txt").write_text("Lesson content.", encoding="utf-8")
    assert index.index_documents_directory(docs) == 1
    assert index.index_documents_directory(docs, force_reindex=True) == 1
    assert len(embeddings.batches) == 2
    assert index.get_stats()["total_documents"] == 1
    assert index.index_documents_directory(docs) == 0


@pytest.mark.asyncio
async def test_single_file_upload_shares_the_directory_manifest(indexing, monkeypatch):
    from rag import query

    index, _, docs, _ = indexing
    file_path = docs / "lesson.txt"
    file_path.write_text("Lesson content.", encoding="utf-8")
    monkeypatch.setattr(query, "vector_index", index)
    first = await query.add_document_to_knowledge_base(str(file_path))
    second = await query.add_document_to_knowledge_base(str(file_path))

    assert first["success"] is True
    assert first["chunks"] == 1
    assert second["status"] == "unchanged"
    assert second["chunks"] == 0
    assert index.index_documents_directory(docs) == 0

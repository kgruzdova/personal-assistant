"""
Vector Index for RAG.
Creates and manages embeddings using ChromaDB.
"""

from typing import List, Optional
from pathlib import Path
import hashlib
import json
import os
import tempfile
from filelock import FileLock
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    DATA_DIR, OPENAI_API_KEY, OPENAI_BASE_URL, DOCUMENTS_DIR,
    RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, PDF_OCR_LANGUAGE,
)
from utils.logging import logger
from rag.loader import document_loader
from rag.pdf_ocr import PdfOcrError


class VectorIndex:
    """Manages vector embeddings and similarity search."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        """
        Initialize vector index.
        
        Args:
            persist_directory: Directory to persist embeddings
        """
        if persist_directory is None:
            persist_directory = DATA_DIR / "chroma_db"
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.persist_directory / "indexed_files.json"
        self._index_lock = FileLock(str(self.persist_directory / "index.lock"))
        self.last_indexing_summary = {}
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL,
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Load existing vectorstore or create new one."""
        try:
            # Try to load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info("Loaded existing vector store")
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore: {e}")
            # Create new vectorstore
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            logger.info("Created new vector store")
    
    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document chunks
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return
            
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> List:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            raise
    
    def index_documents_directory(
        self,
        directory: Path = DOCUMENTS_DIR,
        force_reindex: bool = False
    ) -> int:
        """
        Index new or changed files; skip files already stored with the same hash.
        
        Args:
            directory: Directory containing documents
            force_reindex: If True, clear existing index first
        
        Returns:
            Number of new or updated chunks (not the total index size).
        """
        directory = Path(directory).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        with self._index_lock:
            if force_reindex:
                self.clear_index()
            files = sorted(
                path for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
            )
            return self._sync_files(files)

    def index_document(self, file_path: Path) -> int:
        """Synchronize one file, including calls made outside startup indexing."""
        file_path = Path(file_path).resolve()
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        if file_path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        with self._index_lock:
            return self._sync_files([file_path], raise_errors=True)

    @staticmethod
    def _file_key(file_path: Path) -> str:
        return os.path.normcase(str(Path(file_path).resolve()))

    @staticmethod
    def _content_hash(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _index_signature(self, file_path: Optional[Path] = None) -> str:
        """Changes to splitting or embedding settings also invalidate cached files."""
        settings = {
            "version": 1,
            "chunk_size": RAG_CHUNK_SIZE,
            "chunk_overlap": RAG_CHUNK_OVERLAP,
            "embedding_model": getattr(self.embeddings, "model", None),
            "embedding_dimensions": getattr(self.embeddings, "dimensions", None),
        }
        if file_path is not None and file_path.suffix.lower() == ".pdf":
            settings["pdf_extraction"] = "text-with-windows-ocr-v1"
            # Version 1 already implies Russian; record an override explicitly.
            if PDF_OCR_LANGUAGE != "ru":
                settings["pdf_ocr_language"] = PDF_OCR_LANGUAGE
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def _read_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
            if manifest.get("version") != 1 or not isinstance(files, dict):
                raise ValueError("Unsupported index manifest")
            for key, record in files.items():
                if (not isinstance(key, str) or not isinstance(record, dict)
                        or not isinstance(record.get("sha256"), str)
                        or not isinstance(record.get("signature"), str)
                        or not isinstance(record.get("ids"), list)
                        or not all(isinstance(item, str) for item in record["ids"])):
                    raise ValueError("Invalid index manifest record")
            return files
        except (ValueError, KeyError, TypeError):
            logger.warning("Invalid index manifest; rebuilding file records while preserving stored chunks")
            return {}

    def _write_manifest(self, files: dict):
        """Replace the manifest atomically only after the file update succeeds."""
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.persist_directory,
                prefix=".index-manifest-", suffix=".tmp", delete=False,
            ) as target:
                temporary_path = Path(target.name)
                json.dump({"version": 1, "files": files}, target, ensure_ascii=False, indent=2)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, self.manifest_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _stored_file_chunks(self) -> dict:
        """Include old records without hashes so their duplicates can be replaced."""
        stored = self.vectorstore.get(include=["metadatas"])
        grouped = {}
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"]):
            metadata = metadata or {}
            key = metadata.get("rag_file_key")
            if not key and metadata.get("file_path"):
                key = self._file_key(Path(metadata["file_path"]))
            if key:
                grouped.setdefault(key, []).append((chunk_id, metadata))
        return grouped

    def _sync_files(self, files: List[Path], raise_errors: bool = False) -> int:
        manifest = self._read_manifest()
        stored = self._stored_file_chunks()
        summary = {"new": 0, "updated": 0, "skipped": 0, "failed": 0, "chunks": 0, "removed_chunks": 0}
        self.last_indexing_summary = summary
        for file_path in files:
            try:
                key = self._file_key(file_path)
                signature = self._index_signature(file_path)
                digest = self._content_hash(file_path)
                previous = manifest.get(key)
                old_chunks = stored.get(key, [])
                old_ids = {chunk_id for chunk_id, _ in old_chunks}
                if (previous and previous["sha256"] == digest
                        and (previous["ids"] or file_path.suffix.lower() != ".pdf")
                        and previous["signature"] == signature
                        and set(previous["ids"]) == old_ids
                        and all(metadata.get("rag_sha256") == digest
                                and metadata.get("rag_signature") == signature
                                for _, metadata in old_chunks)):
                    summary["skipped"] += 1
                    logger.info("Unchanged file skipped: %s", file_path.name)
                    continue

                chunks = document_loader.load_document(file_path)
                if not chunks and file_path.suffix.lower() == ".pdf":
                    raise PdfOcrError("PDF text extraction returned zero fragments; the file will be retried")
                if self._content_hash(file_path) != digest:
                    raise RuntimeError("File changed while being read; retry on next indexing")
                ids = []
                for position, chunk in enumerate(chunks):
                    chunk_id = hashlib.sha256(
                        json.dumps([key, digest, signature, position]).encode()
                    ).hexdigest()
                    ids.append(chunk_id)
                    chunk.metadata.update({
                        "file_path": str(file_path), "rag_file_key": key,
                        "rag_sha256": digest, "rag_signature": signature,
                    })

                # Add first: failed embeddings must not delete the working version.
                # Stable IDs make a retry after partial writes an upsert, not a duplicate.
                if chunks:
                    self.vectorstore.add_documents(chunks, ids=ids)
                obsolete = sorted(old_ids - set(ids))
                if obsolete:
                    self.vectorstore.delete(ids=obsolete)
                manifest[key] = {"sha256": digest, "signature": signature, "ids": ids}
                self._write_manifest(manifest)
                status = "updated" if previous is not None or old_ids else "new"
                summary[status] += 1
                summary["chunks"] += len(chunks)
                summary["removed_chunks"] += len(obsolete)
                logger.info("File %s: %s (%s chunks, %s old chunks removed)", status, file_path.name, len(chunks), len(obsolete))
            except Exception as error:
                summary["failed"] += 1
                logger.error("Could not index %s: %s", file_path.name, str(error) if isinstance(error, PdfOcrError) else type(error).__name__)
                if raise_errors:
                    raise
        logger.info(
            "Index scan completed: new=%s, updated=%s, skipped=%s, failed=%s, chunks=%s",
            summary["new"], summary["updated"], summary["skipped"], summary["failed"], summary["chunks"],
        )
        return summary["chunks"]

    def clear_index(self):
        """Clear the entire vector store."""
        with self._index_lock:
            # Chroma owns its open database files; clear through its API on Windows too.
            self.vectorstore.delete_collection()
            self.manifest_path.unlink(missing_ok=True)
            self._load_or_create_vectorstore()
            logger.info("Vector store cleared")
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # ChromaDB collection stats
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}


# Global index instance
vector_index = VectorIndex()


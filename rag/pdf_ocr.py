"""Local OCR for PDF pages without a text layer, using Windows built-in APIs."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from filelock import FileLock

from config import DATA_DIR, PDF_OCR_LANGUAGE
from utils.logging import logger


OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
OCR_VERSION = 1


class PdfOcrError(ValueError):
    """A PDF page could not be read reliably for indexing."""


def _pdf_hash(file_path: Path) -> str:
    with file_path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _validate_pages(rows, requested):
    if not isinstance(rows, list):
        raise PdfOcrError("OCR returned an invalid page list")
    pages = {}
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("page")) is not int
                or not isinstance(row.get("text"), str) or row["page"] in pages):
            raise PdfOcrError("OCR returned an invalid page record")
        pages[row["page"]] = row
    if not set(requested).issubset(pages):
        raise PdfOcrError("OCR did not process all requested pages")
    return {page: pages[page] for page in requested}


def _run_windows_ocr(file_path: Path, page_numbers: list, output_path: Path):
    if sys.platform != "win32":
        raise PdfOcrError("PDF has no text layer. Automatic OCR requires Windows; on this OS, add a searchable text layer with an OCR tool first.")
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not powershell.is_file():
        raise PdfOcrError("Windows PowerShell is required for local PDF OCR")
    script = Path(__file__).with_name("windows_pdf_ocr.ps1")
    try:
        result = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(script), "-InputPath", str(file_path), "-OutputPath", str(output_path),
             "-PageNumbers", ",".join(map(str, page_numbers)), "-Language", PDF_OCR_LANGUAGE],
            capture_output=True, timeout=max(90, 60 + 20 * len(page_numbers)),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as error:
        raise PdfOcrError("Local PDF OCR timed out; the file will be retried on the next indexing") from error
    if result.returncode != 0 or not output_path.is_file():
        details = result.stderr.decode("utf-8", errors="replace")[-1600:]
        raise PdfOcrError(f"Windows PDF OCR failed. Check the installed OCR language '{PDF_OCR_LANGUAGE}'. {details}")


def recognize_pdf_pages(file_path: Path, page_numbers: list) -> dict:
    """Return OCR text by zero-based page number; cache it by file content hash."""
    file_path = Path(file_path).resolve()
    page_numbers = sorted(set(page_numbers))
    if not page_numbers:
        return {}
    digest = _pdf_hash(file_path)
    cache_key = hashlib.sha256(json.dumps([digest, PDF_OCR_LANGUAGE, OCR_VERSION]).encode()).hexdigest()
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OCR_CACHE_DIR / f"{cache_key}.json"
    with FileLock(str(cache_path.with_suffix(".lock"))):
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if (cached.get("version") != OCR_VERSION or cached.get("sha256") != digest
                        or cached.get("language") != PDF_OCR_LANGUAGE
                        or not any(row.get("text", "").strip() for row in cached.get("pages", []) if isinstance(row, dict))):
                    raise ValueError("OCR cache does not match this document")
                return _validate_pages(cached["pages"], page_numbers)
            except (ValueError, KeyError, TypeError):
                logger.warning("Invalid or incomplete OCR cache; recognizing PDF again: %s", file_path.name)
        logger.info("Recognizing %s PDF pages locally with Windows OCR: %s", len(page_numbers), file_path.name)
        with tempfile.TemporaryDirectory(prefix="rag-pdf-ocr-") as work_dir:
            output_path = Path(work_dir) / "pages.json"
            _run_windows_ocr(file_path, page_numbers, output_path)
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            pages = _validate_pages(rows, page_numbers)
            if _pdf_hash(file_path) != digest:
                raise PdfOcrError("PDF changed during OCR; retry indexing")
            # A genuinely blank page is valid in a mixed PDF, but do not cache
            # an entirely empty OCR result as a successful extraction.
            if not any(row["text"].strip() for row in pages.values()):
                return pages
            cache = {"version": OCR_VERSION, "sha256": digest, "language": PDF_OCR_LANGUAGE, "pages": list(pages.values())}
            temporary = cache_path.with_suffix(".tmp")
            try:
                temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                os.replace(temporary, cache_path)
            finally:
                temporary.unlink(missing_ok=True)
        logger.info("OCR completed for %s: %s pages, %s characters", file_path.name, len(pages), sum(len(row["text"]) for row in pages.values()))
        return pages

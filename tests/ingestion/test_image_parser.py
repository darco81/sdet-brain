"""parse_image / parse_pdf — exercised with a fake OCR engine + fake pdfium."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pypdfium2
import pytest
from PIL import Image

from sdet_brain.config import Settings
from sdet_brain.ingestion.image_parser import (
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    is_image_path,
    is_pdf_path,
    parse_image,
    parse_pdf,
)
from sdet_brain.ingestion.models import ParsedDocument
from sdet_brain.ocr.protocol import OCRError, OCRResult


class _FakeOCREngine:
    def __init__(
        self,
        *,
        text: str = "Receipt total 24,99 PLN with enough text.",
        model: str = "fake-model",
        peak_memory_gb: float | None = None,
    ) -> None:
        self._text = text
        self._model = model
        self._peak = peak_memory_gb
        self.calls: list[bytes] = []

    @property
    def model_name(self) -> str:
        return self._model

    def extract_text(
        self, image_bytes: bytes, *, prompt: str | None = None
    ) -> OCRResult:
        _ = prompt
        self.calls.append(image_bytes)
        return OCRResult(
            text=self._text,
            model=self._model,
            duration_s=0.1,
            peak_memory_gb=self._peak,
        )

    def health_check(self) -> bool:
        return True


def _make_png(path: Path, *, width: int = 50, height: int = 50) -> None:
    img = Image.new("RGB", (width, height), color="white")
    img.save(path, format="PNG")


# --- parse_image -----------------------------------------------------------


def test_parse_image_returns_parsed_document(tmp_path: Path) -> None:
    path = tmp_path / "receipt.png"
    _make_png(path)
    engine = _FakeOCREngine(text="Receipt total 99,99 PLN paid on 14/05.")
    settings = Settings()

    doc = parse_image(path, ocr_engine=engine, settings=settings)

    assert isinstance(doc, ParsedDocument)
    assert doc.source_path == str(path)
    assert len(doc.content_hash) == 64  # sha256 hex
    assert doc.frontmatter["source"] == "ocr"
    assert doc.frontmatter["source_type"] == "image-ocr"
    assert doc.frontmatter["ocr_model"] == "fake-model"
    assert doc.frontmatter["original_extension"] == ".png"
    assert "extracted_at" in doc.frontmatter
    assert len(doc.chunks) >= 1
    assert "Receipt total" in doc.chunks[0].text


def test_parse_image_size_cap_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "huge.png"
    _make_png(path, width=400, height=400)  # ~few KB regardless
    settings = Settings(ocr_max_image_bytes=10)  # absurdly small cap

    with pytest.raises(OCRError, match="exceeds"):
        parse_image(path, ocr_engine=_FakeOCREngine(), settings=settings)


def test_parse_image_resizes_oversized_dimensions(tmp_path: Path) -> None:
    """Engine should receive normalized bytes whose long edge ≤ max_dim."""
    path = tmp_path / "big.png"
    _make_png(path, width=3000, height=2000)
    engine = _FakeOCREngine()
    settings = Settings(ocr_max_image_dim=512)

    parse_image(path, ocr_engine=engine, settings=settings)

    assert len(engine.calls) == 1
    with Image.open(io.BytesIO(engine.calls[0])) as decoded:
        assert max(decoded.size) <= 512


def test_parse_image_content_hash_uses_raw_bytes(tmp_path: Path) -> None:
    """Idempotency hinges on the file-bytes hash, NOT the normalized PNG."""
    path = tmp_path / "stable.png"
    _make_png(path)
    raw = path.read_bytes()

    import hashlib

    doc = parse_image(
        path, ocr_engine=_FakeOCREngine(), settings=Settings(),
    )

    assert doc.content_hash == hashlib.sha256(raw).hexdigest()


# --- parse_pdf -------------------------------------------------------------


class _FakeBitmap:
    def to_pil(self) -> Image.Image:
        return Image.new("RGB", (200, 200), color="white")


class _FakePage:
    def render(self, scale: float = 1.0) -> _FakeBitmap:
        _ = scale
        return _FakeBitmap()


class _FakePdfDocument:
    def __init__(self, _path: Any, pages: int = 2) -> None:
        self._pages = pages
        self.closed = False

    def __len__(self) -> int:
        return self._pages

    def __getitem__(self, index: int) -> _FakePage:
        _ = index
        return _FakePage()

    def close(self) -> None:
        self.closed = True


def _patch_pdfium(
    monkeypatch: pytest.MonkeyPatch, *, pages: int = 2
) -> None:
    monkeypatch.setattr(
        pypdfium2,
        "PdfDocument",
        lambda path: _FakePdfDocument(path, pages=pages),
    )


def test_parse_pdf_concatenates_pages_with_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4 fake header")
    _patch_pdfium(monkeypatch, pages=2)
    engine = _FakeOCREngine(text="page body text long enough for chunker")
    settings = Settings()

    doc = parse_pdf(path, ocr_engine=engine, settings=settings)

    assert doc.frontmatter["total_pages"] == 2
    assert doc.frontmatter["original_extension"] == ".pdf"
    full_text = "\n\n".join(c.text for c in doc.chunks)
    assert "## Page 1" in full_text
    assert "## Page 2" in full_text
    assert len(engine.calls) == 2  # one OCR call per page


def test_parse_pdf_rejects_page_count_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "huge.pdf"
    path.write_bytes(b"%PDF fake")
    _patch_pdfium(monkeypatch, pages=99)
    settings = Settings(ocr_max_pdf_pages=5)

    with pytest.raises(OCRError, match="exceeds"):
        parse_pdf(path, ocr_engine=_FakeOCREngine(), settings=settings)


def test_parse_pdf_closes_document_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF fake")

    instances: list[_FakePdfDocument] = []

    def factory(p: Any) -> _FakePdfDocument:
        doc = _FakePdfDocument(p, pages=2)
        instances.append(doc)
        return doc

    monkeypatch.setattr(pypdfium2, "PdfDocument", factory)

    class _BoomEngine:
        @property
        def model_name(self) -> str:
            return "boom"

        def extract_text(
            self, image_bytes: bytes, *, prompt: str | None = None
        ) -> OCRResult:
            _ = image_bytes
            _ = prompt
            raise OCRError("simulated OCR failure")

        def health_check(self) -> bool:
            return True

    with pytest.raises(OCRError, match="simulated"):
        parse_pdf(path, ocr_engine=_BoomEngine(), settings=Settings())

    assert len(instances) == 1
    assert instances[0].closed is True


# --- helpers --------------------------------------------------------------


def test_is_image_path_recognises_common_suffixes() -> None:
    assert is_image_path(Path("a.jpg"))
    assert is_image_path(Path("b.HEIC"))
    assert is_image_path(Path("c.png"))
    assert not is_image_path(Path("d.pdf"))
    assert not is_image_path(Path("e.md"))


def test_is_pdf_path_case_insensitive() -> None:
    assert is_pdf_path(Path("scan.pdf"))
    assert is_pdf_path(Path("SCAN.PDF"))
    assert not is_pdf_path(Path("image.jpg"))


def test_suffix_sets_are_disjoint() -> None:
    assert not (IMAGE_SUFFIXES & PDF_SUFFIXES)

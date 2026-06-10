"""Image / PDF ingestion via OCR.

``parse_image`` and ``parse_pdf`` accept a filesystem path plus an
``IOCREngine`` and emit a :class:`ParsedDocument` whose chunks carry
the OCR'd markdown text. The shapes match :func:`parse_markdown` so
the rest of the pipeline (chunking, embedding, Qdrant upsert) is
unchanged.

Preprocessing pipeline before OCR:

1. Size guard against ``OCR_MAX_IMAGE_BYTES`` / ``OCR_MAX_PDF_PAGES``
   (edge cases #3, #4 in the v0.6.0 plan).
2. EXIF transpose so phone photos OCR upright (edge case #1).
3. Resize the long edge to ``OCR_MAX_IMAGE_DIM`` — bigger inputs cost
   VRAM without quality gains for receipt-grade text.
4. Re-encode as PNG so providers see a stable, lossless container
   regardless of the input format (JPEG, HEIC, etc.).

HEIC support is registered globally on module import via
``pillow-heif``. PDF page rendering uses ``pypdfium2`` (pure-pip, no
Poppler binary required).
"""

from __future__ import annotations

import hashlib
import io
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from PIL import Image, ImageOps, UnidentifiedImageError

from sdet_brain.ingestion.chunker import (
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_TARGET_CHARS,
    chunk_markdown,
)
from sdet_brain.ingestion.models import ParsedDocument
from sdet_brain.ocr.protocol import IOCREngine, OCRError, OCRResult

if TYPE_CHECKING:
    from sdet_brain.config import Settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tiff", ".tif"},
)
"""Filesystem suffixes routed to :func:`parse_image`."""

PDF_SUFFIXES: Final[frozenset[str]] = frozenset({".pdf"})
"""Filesystem suffixes routed to :func:`parse_pdf`."""

_PDF_RENDER_SCALE: Final[float] = 2.0
"""Render multiplier for PDF pages — ~144 DPI at default 72 DPI base."""


# Register HEIC/HEIF opener once at import. pillow-heif is in deps but we
# guard the import so non-Apple builds (where HEIC is rare) still load.
try:
    import pillow_heif as _pillow_heif

    _pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - cross-platform safety net
    logger.debug("pillow-heif unavailable; HEIC ingestion disabled")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Coerce any mode into RGB without losing alpha to a black void.

    ``RGB`` / ``L`` pass through. ``RGBA`` / ``LA`` / palette-with-alpha
    are composited onto a white background — receipts photographed
    against a checkered preview background OCR worse if alpha leaks
    through as black. Everything else (``P``, ``CMYK``, ``I``, ``F``)
    falls through to plain ``convert("RGB")``.
    """
    if image.mode in {"RGB", "L"}:
        return image
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return image.convert("RGB")


def _normalize_image_bytes(raw: bytes, *, max_dim: int) -> bytes:
    """Apply EXIF transpose + resize + PNG re-encode.

    Wraps PIL exceptions into :class:`OCRError` with actionable
    messages so callers can distinguish "bad input file" from
    "OCR backend dead".
    """
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            transposed = ImageOps.exif_transpose(opened) or opened
            if max(transposed.size) > max_dim:
                transposed.thumbnail(
                    (max_dim, max_dim),
                    Image.Resampling.LANCZOS,
                )
            transposed = _flatten_to_rgb(transposed)
            buf = io.BytesIO()
            transposed.save(buf, format="PNG")
            return buf.getvalue()
    except UnidentifiedImageError as exc:
        raise OCRError(
            "Cannot decode image — unsupported format or corrupt file. "
            "Try re-exporting as PNG/JPEG.",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise OCRError(
            "Image is suspiciously large (PIL decompression-bomb guard). "
            "Resize the source below ~90 MP before ingesting.",
        ) from exc


def _frontmatter(
    path: Path,
    result: OCRResult,
    *,
    page_number: int | None,
    total_pages: int | None,
) -> dict[str, object]:
    fm: dict[str, object] = {
        "source": "ocr",
        "source_type": "image-ocr",
        "ocr_model": result.model,
        "extracted_at": datetime.now(tz=UTC).isoformat(),
        "original_path": str(path),
        "original_extension": path.suffix.lower(),
    }
    if page_number is not None:
        fm["page_number"] = page_number
    if total_pages is not None:
        fm["total_pages"] = total_pages
    if result.peak_memory_gb is not None:
        fm["ocr_peak_memory_gb"] = result.peak_memory_gb
    fm["ocr_duration_s"] = result.duration_s
    return fm


def parse_image(
    path: Path,
    *,
    ocr_engine: IOCREngine,
    settings: Settings,
) -> ParsedDocument:
    """OCR a single image into a :class:`ParsedDocument`.

    Raises :class:`OCRError` when the image exceeds the configured size
    cap or cannot be decoded.
    """
    size = path.stat().st_size
    if size > settings.ocr_max_image_bytes:
        raise OCRError(
            f"Image {path.name} is {size} bytes; exceeds "
            f"OCR_MAX_IMAGE_BYTES={settings.ocr_max_image_bytes}.",
        )

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()
    normalized = _normalize_image_bytes(raw, max_dim=settings.ocr_max_image_dim)

    result = ocr_engine.extract_text(normalized)
    chunks = chunk_markdown(
        result.text,
        target_size=settings.chunk_target_chars,
        overlap_pct=settings.chunk_overlap_ratio,
    )
    return ParsedDocument(
        source_path=str(path),
        content_hash=content_hash,
        frontmatter=_frontmatter(
            path,
            result,
            page_number=None,
            total_pages=None,
        ),
        chunks=tuple(chunks),
    )


def _render_pdf_page_png(page: object, *, max_dim: int) -> bytes:
    """Render a pypdfium2 page to a PNG byte string at the configured scale."""
    bitmap = page.render(scale=_PDF_RENDER_SCALE)  # type: ignore[attr-defined]
    pil_image = bitmap.to_pil()
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    pil_image = _flatten_to_rgb(pil_image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def parse_pdf(
    path: Path,
    *,
    ocr_engine: IOCREngine,
    settings: Settings,
) -> ParsedDocument:
    """OCR every page of a PDF and concatenate into one document.

    Page boundaries survive as ``## Page N`` markdown headings inside
    the chunker output; per-chunk page tracking is deferred to v0.6.1
    (would require widening :class:`Chunk`).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dep is mandatory
        raise OCRError(
            "pypdfium2 not installed — cannot ingest PDFs. Run `uv sync`.",
        ) from exc

    raw = path.read_bytes()
    content_hash = hashlib.sha256(raw).hexdigest()

    pdf = pdfium.PdfDocument(path)
    try:
        total_pages = len(pdf)
        if total_pages > settings.ocr_max_pdf_pages:
            raise OCRError(
                f"PDF {path.name} has {total_pages} pages; exceeds "
                f"OCR_MAX_PDF_PAGES={settings.ocr_max_pdf_pages}.",
            )

        page_blocks: list[str] = []
        last_result: OCRResult | None = None
        for page_index in range(total_pages):
            page = pdf[page_index]
            png_bytes = _render_pdf_page_png(
                page,
                max_dim=settings.ocr_max_image_dim,
            )
            result = ocr_engine.extract_text(png_bytes)
            last_result = result
            page_blocks.append(f"## Page {page_index + 1}\n\n{result.text}")
    finally:
        pdf.close()

    if last_result is None:  # zero-page PDF — pdfium would normally reject earlier
        raise OCRError(f"PDF {path.name} appears empty (no pages rendered).")

    full_text = "\n\n".join(page_blocks)
    chunks = chunk_markdown(
        full_text,
        target_size=settings.chunk_target_chars,
        overlap_pct=settings.chunk_overlap_ratio,
    )
    return ParsedDocument(
        source_path=str(path),
        content_hash=content_hash,
        frontmatter=_frontmatter(
            path,
            last_result,
            page_number=None,
            total_pages=total_pages,
        ),
        chunks=tuple(chunks),
    )


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_pdf_path(path: Path) -> bool:
    return path.suffix.lower() in PDF_SUFFIXES


# Re-export DEFAULT_TARGET_CHARS / DEFAULT_OVERLAP_RATIO consumers like.
__all__ = [
    "DEFAULT_OVERLAP_RATIO",
    "DEFAULT_TARGET_CHARS",
    "IMAGE_SUFFIXES",
    "PDF_SUFFIXES",
    "is_image_path",
    "is_pdf_path",
    "parse_image",
    "parse_pdf",
]

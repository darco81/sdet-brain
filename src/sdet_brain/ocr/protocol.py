"""OCR provider contract shared by MLX-VLM, Ollama, and future backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class OCRResult:
    """Output of a single OCR call against one image (or PDF page).

    PDF page-number metadata is set by the parser layer; the engine
    sees one image at a time and reports only what it produced.
    """

    text: str
    """Markdown text extracted from the image (grounding-tokens already stripped)."""

    model: str
    """Identifier of the model that produced ``text``."""

    duration_s: float
    """Wall-clock seconds spent on the OCR call."""

    peak_memory_gb: float | None = None
    """Peak GPU / unified memory in GB during the call.

    ``None`` when the backend cannot measure (e.g. remote Ollama).
    """


@runtime_checkable
class IOCREngine(Protocol):
    """Provider-agnostic OCR interface.

    Implementations transform image bytes into markdown text. The
    protocol is runtime-checkable so the factory can verify a
    candidate engine before wiring it up.
    """

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``mlx-community/DeepSeek-OCR-2-6bit``)."""
        ...

    def extract_text(self, image_bytes: bytes, *, prompt: str | None = None) -> OCRResult:
        """Run OCR against a single in-memory image.

        ``image_bytes`` should be a fully-decoded raster image (JPEG/PNG/...).
        EXIF rotation, HEIC conversion, and resize-to-budget are the
        caller's responsibility — engines accept the bytes as-is.

        ``prompt`` overrides the engine's default OCR prompt. Pass
        ``None`` to use the provider's standard grounding prompt.
        """
        ...

    def health_check(self) -> bool:
        """Return ``True`` if the engine can OCR a probe image."""
        ...


class OCRError(Exception):
    """Raised when an OCR provider cannot produce text.

    Providers MUST wrap their transport-layer errors (``httpx.ConnectError``,
    ``ImportError`` from missing optional deps, etc.) into ``OCRError``
    before raising — the factory's fallback chain catches ``OCRError``
    only and lets unrelated exceptions bubble. See
    :func:`sdet_brain.ocr.factory._try_build`.

    Inherits from ``Exception`` (not ``RuntimeError``) because OCR
    failure is a domain error, not an interpreter-classification gap.
    """


class OCRTimeoutError(OCRError):
    """Raised when an OCR call exceeds ``ocr_timeout_seconds``."""


class OCRQualityError(OCRError):
    """Raised when OCR output fails the quality heuristic.

    Triggers the factory to fall back to the next engine in the chain.
    Heuristic triggers (see plan §Quality-fallback):

    * output shorter than ``ocr_quality_min_chars`` after token strip,
    * output empty after grounding-token strip,
    * output is a single line of whitespace / punctuation only.
    """

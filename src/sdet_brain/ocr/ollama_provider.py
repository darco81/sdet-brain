"""Ollama-backed OCR provider — cross-platform fallback.

Supports any vision model served by an Ollama daemon: ``deepseek-ocr``
(DeepSeek-OCR with grounding tokens), ``qwen2.5-vl:32b`` (heavyweight
multilingual), and anything else with vision support. The model tag
flows in from the factory; the engine treats it as opaque.

Payload format: ``POST /api/generate`` with base64-encoded image bytes,
``stream=false``, and the user-tuned ``keep_alive`` directive that
unloads idle weights after the configured idle window (edge case #10
in the v0.6.0 plan — protects 4 GB VRAM on the Win flagship).

The provider does NOT expose peak-memory metrics — Ollama's response
doesn't report VRAM usage. The OCRResult therefore leaves
``peak_memory_gb=None``.
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Any, Final

import httpx

from sdet_brain.ocr.prompts import (
    deduplicate_repeats,
    quality_acceptable,
    strip_deepseek_tokens,
)
from sdet_brain.ocr.protocol import (
    OCRError,
    OCRQualityError,
    OCRResult,
    OCRTimeoutError,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST: Final[str] = "http://localhost:11434"
_HEALTH_TIMEOUT_S: Final[float] = 2.0


class OllamaOCREngine:
    """OCR engine that drives a vision model via the Ollama HTTP API.

    Implements :class:`sdet_brain.ocr.protocol.IOCREngine`. Stateless
    aside from configuration — every call opens a short-lived
    ``httpx`` request so process restarts don't strand connections.
    """

    def __init__(
        self,
        *,
        model_name: str,
        default_prompt: str,
        quality_min_chars: int,
        keep_alive: str = "5m",
        timeout_seconds: int = 120,
        host: str = DEFAULT_HOST,
    ) -> None:
        self._model_tag = model_name
        self._default_prompt = default_prompt
        self._quality_min_chars = quality_min_chars
        self._keep_alive = keep_alive
        self._timeout_seconds = timeout_seconds
        self._host = host.rstrip("/")

    @property
    def model_name(self) -> str:
        """Public identifier — namespaced so payloads can distinguish backends."""
        return f"ollama:{self._model_tag}"

    def health_check(self) -> bool:
        """Verify the Ollama daemon answers at ``{host}/api/tags``.

        Logs the exception itself (not just the URL) so the operator
        can distinguish "daemon down" (``ConnectError``) from
        "wrong proxy" (``ProxyError``) from "daemon up, returned 5xx"
        (``HTTPStatusError``).
        """
        try:
            response = httpx.get(
                f"{self._host}/api/tags", timeout=_HEALTH_TIMEOUT_S,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "Ollama health check failed at %s/api/tags: %s",
                self._host,
                exc,
            )
            return False
        except httpx.InvalidURL as exc:
            # InvalidURL is a subclass of Exception, NOT HTTPError —
            # a malformed OCR_OLLAMA_HOST env var would otherwise crash
            # the factory boot. Surface as a normal health failure so
            # the chain can fall back.
            logger.warning(
                "Ollama host %r is not a valid URL: %s", self._host, exc,
            )
            return False
        return True

    def extract_text(
        self, image_bytes: bytes, *, prompt: str | None = None
    ) -> OCRResult:
        if not image_bytes:
            raise OCRError("Empty image_bytes — nothing to OCR.")

        effective_prompt = prompt if prompt is not None else self._default_prompt
        payload: dict[str, Any] = {
            "model": self._model_tag,
            "prompt": effective_prompt,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"temperature": 0.0},
        }

        t0 = time.time()
        try:
            response = httpx.post(
                f"{self._host}/api/generate",
                json=payload,
                timeout=float(self._timeout_seconds),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise OCRTimeoutError(
                f"Ollama OCR call exceeded {self._timeout_seconds}s "
                f"(model={self._model_tag!r}).",
            ) from exc
        except httpx.HTTPError as exc:
            raise OCRError(
                f"Ollama OCR request failed (model={self._model_tag!r}): {exc}",
            ) from exc
        elapsed = time.time() - t0

        try:
            data = response.json()
        except ValueError as exc:
            raise OCRError(
                f"Ollama returned non-JSON response (model={self._model_tag!r}).",
            ) from exc

        raw_text = str(data.get("response", ""))
        cleaned = deduplicate_repeats(strip_deepseek_tokens(raw_text))

        if not quality_acceptable(cleaned, min_chars=self._quality_min_chars):
            raise OCRQualityError(
                f"Ollama OCR output below quality bar "
                f"(model={self._model_tag!r}, "
                f"chars={len(cleaned.strip())}, "
                f"min={self._quality_min_chars}).",
            )

        return OCRResult(
            text=cleaned,
            model=self.model_name,
            duration_s=round(elapsed, 2),
            peak_memory_gb=None,
        )

"""MLX-VLM OCR provider — DeepSeek-OCR on Apple Silicon.

DeepSeek-OCR-2-6bit (3B parameters, ~5 GB peak RAM) runs ~3 s per
receipt on an M-series Mac, roughly 14x faster than the equivalent
Qwen2.5-VL Ollama path. Loads lazily on the first ``extract_text``
call so module import stays cheap.

Ported from the production-validated Domowy Kombajn engine
(`m5_service/ocr_engine.py:1-107`). The MLX arena hygiene
(``mlx.core.clear_cache()`` in finally) mirrors the embedding-side
pattern (`src/sdet_brain/embeddings/mlx_provider.py:120-126`).
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from sdet_brain.ocr.prompts import (
    deduplicate_repeats,
    quality_acceptable,
    strip_deepseek_tokens,
)
from sdet_brain.ocr.protocol import OCRError, OCRQualityError, OCRResult

logger = logging.getLogger(__name__)


class MLXVLMOCREngine:
    """OCR engine backed by MLX-VLM (Apple Silicon only).

    Implements :class:`sdet_brain.ocr.protocol.IOCREngine`. The model
    is loaded on the first ``extract_text`` call (double-checked lock)
    and re-used for the lifetime of the engine. The factory's
    process-level singleton ensures only one engine per process.
    """

    def __init__(
        self,
        *,
        model_name: str,
        default_prompt: str,
        quality_min_chars: int,
    ) -> None:
        self._model_name = model_name
        self._default_prompt = default_prompt
        self._quality_min_chars = quality_min_chars
        self._lock = threading.Lock()
        # mlx-vlm publishes no py.typed stubs; everything stays Any.
        self._model: Any | None = None
        self._processor: Any | None = None
        self._config: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def health_check(self) -> bool:
        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            logger.warning(
                "mlx-vlm not installed; MLX-VLM OCR provider unavailable on this host.",
            )
            return False
        return True

    def _load_model(self) -> tuple[Any, Any, Any]:
        """Import mlx-vlm submodules and load weights.

        Split out so tests can monkeypatch without touching real MLX.
        """
        # CRITICAL: the explicit submodule import registers
        # ``DeepseekOCR2Processor`` and monkey-patches
        # ``AutoProcessor.from_pretrained``. ``from mlx_vlm import load``
        # alone does NOT trigger that registration.
        import mlx_vlm.models.deepseekocr_2.processing_deepseekocr  # noqa: F401
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        logger.info("Loading MLX-VLM model %s (first call, ~5-10 s)", self._model_name)
        t0 = time.time()
        # trust_remote_code=False — the local patched processor is used
        # instead of pulling HF remote code (which would drag in
        # torch+addict+einops just for modeling_*.py).
        model, processor = load(self._model_name, trust_remote_code=False)
        config = load_config(self._model_name)
        logger.info("MLX-VLM model loaded in %.1f s", time.time() - t0)
        return model, processor, config

    def _ensure_loaded(self) -> tuple[Any, Any, Any]:
        if (
            self._model is not None
            and self._processor is not None
            and self._config is not None
        ):
            return self._model, self._processor, self._config
        with self._lock:
            # Re-check all three under lock: a half-built engine (e.g.
            # _load_model raised after model + processor loaded but
            # before config returned) leaves _model set while _config
            # is None. Checking only _model would let the next call
            # short-circuit on a partial state.
            if (
                self._model is None
                or self._processor is None
                or self._config is None
            ):
                try:
                    self._model, self._processor, self._config = self._load_model()
                except ImportError as exc:
                    raise OCRError(
                        f"mlx-vlm not available — cannot load {self._model_name!r}. "
                        "Install with `uv sync` on Apple Silicon.",
                    ) from exc
        assert self._model is not None  # noqa: S101 - lock guarantees this
        assert self._processor is not None  # noqa: S101
        assert self._config is not None  # noqa: S101
        return self._model, self._processor, self._config

    def _generate(
        self,
        model: Any,
        processor: Any,
        config: Any,
        prompt: str,
        image_path: Path,
    ) -> Any:
        """Invoke ``mlx_vlm.generate`` — split out for testability."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        return generate(
            model,
            processor,
            formatted,
            [str(image_path)],
            max_tokens=2500,
            temperature=0.0,
            verbose=False,
        )

    def extract_text(
        self, image_bytes: bytes, *, prompt: str | None = None
    ) -> OCRResult:
        if not image_bytes:
            raise OCRError("Empty image_bytes — nothing to OCR.")

        model, processor, config = self._ensure_loaded()
        effective_prompt = prompt if prompt is not None else self._default_prompt

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tmp.write(image_bytes)
            tmp.flush()
            tmp_path = Path(tmp.name)

            t0 = time.time()
            try:
                output = self._generate(
                    model, processor, config, effective_prompt, tmp_path
                )
            finally:
                # Release Metal/unified-memory arena that MLX would
                # otherwise carry between calls. Same pattern as
                # embeddings/mlx_provider.py:120-126.
                try:
                    import mlx.core as mx
                except ImportError:
                    # A partial install (mlx_vlm present, mlx.core
                    # missing) is a real config error — let it bubble
                    # so the user knows their venv is broken.
                    logger.exception(
                        "mlx.core missing — MLX install is broken, re-run `uv sync`",
                    )
                    raise
                try:
                    mx.clear_cache()
                except RuntimeError as exc:  # pragma: no cover - documented MLX failure
                    # Documented MLX failure mode — Metal arena likely
                    # leaked but inference already completed.
                    logger.warning(
                        "MLX clear_cache failed (Metal arena likely leaked): %s",
                        exc,
                    )
            elapsed = time.time() - t0

        # mlx-vlm 0.5.x returns either ``GenerationResult`` or raw ``str``.
        if hasattr(output, "text"):
            raw_text = str(output.text)
            peak_mb = float(getattr(output, "peak_memory", 0) or 0)
        else:
            raw_text = str(output)
            peak_mb = 0.0

        cleaned = deduplicate_repeats(strip_deepseek_tokens(raw_text))

        if not quality_acceptable(cleaned, min_chars=self._quality_min_chars):
            raise OCRQualityError(
                f"MLX-VLM output below quality bar "
                f"(model={self._model_name!r}, "
                f"chars={len(cleaned.strip())}, "
                f"min={self._quality_min_chars}).",
            )

        peak_gb = round(peak_mb / 1024, 2) if peak_mb else None
        return OCRResult(
            text=cleaned,
            model=self._model_name,
            duration_s=round(elapsed, 2),
            peak_memory_gb=peak_gb,
        )

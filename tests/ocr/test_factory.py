"""Factory fallback chain — exercised with in-process stubs (no MLX/Ollama)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from sdet_brain.config import Settings
from sdet_brain.ocr import factory as factory_module
from sdet_brain.ocr.factory import (
    OCREngineSelection,
    get_ocr_engine,
    reset_ocr_engine,
)
from sdet_brain.ocr.protocol import OCRError, OCRResult


class _StubOCREngine:
    def __init__(self, *, model_name: str, healthy: bool) -> None:
        self._model_name = model_name
        self._healthy = healthy

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract_text(
        self, image_bytes: bytes, *, prompt: str | None = None
    ) -> OCRResult:
        _ = image_bytes
        _ = prompt
        return OCRResult(text="stub", model=self._model_name, duration_s=0.001)

    def health_check(self) -> bool:
        return self._healthy


@pytest.fixture
def patched_builders(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {
        "mlx_vlm_healthy": True,
        "ollama_primary_healthy": True,
        "ollama_fallback_healthy": True,
    }

    def build_mlx_vlm(_: Settings, model: str) -> _StubOCREngine:
        return _StubOCREngine(model_name=model, healthy=state["mlx_vlm_healthy"])

    def build_ollama(_: Settings, model: str) -> _StubOCREngine:
        # Distinguish primary deepseek-ocr from heavyweight qwen fallback by tag.
        healthy = (
            state["ollama_fallback_healthy"]
            if model.startswith("qwen")
            else state["ollama_primary_healthy"]
        )
        return _StubOCREngine(model_name=f"ollama:{model}", healthy=healthy)

    monkeypatch.setitem(factory_module._BUILDERS, "mlx-vlm", build_mlx_vlm)
    monkeypatch.setitem(factory_module._BUILDERS, "ollama", build_ollama)
    reset_ocr_engine()
    yield state
    reset_ocr_engine()


def _settings(provider: str = "mlx-vlm", **overrides: Any) -> Settings:
    return Settings(ocr_provider=provider, **overrides)  # type: ignore[arg-type]


def test_primary_healthy_returns_primary(patched_builders: dict[str, Any]) -> None:
    selection = get_ocr_engine(_settings("mlx-vlm"))
    assert isinstance(selection, OCREngineSelection)
    assert selection.provider == "mlx-vlm"
    assert selection.model == "mlx-community/DeepSeek-OCR-2-6bit"
    assert selection.fell_back is False
    assert selection.attempted == (("mlx-vlm", "mlx-community/DeepSeek-OCR-2-6bit"),)
    assert selection.engine.model_name == "mlx-community/DeepSeek-OCR-2-6bit"


def test_primary_unhealthy_falls_back_to_secondary(
    patched_builders: dict[str, Any],
) -> None:
    patched_builders["mlx_vlm_healthy"] = False
    selection = get_ocr_engine(_settings("mlx-vlm"))
    assert selection.provider == "ollama"
    assert selection.model == "deepseek-ocr"
    assert selection.fell_back is True
    assert selection.attempted == (
        ("mlx-vlm", "mlx-community/DeepSeek-OCR-2-6bit"),
        ("ollama", "deepseek-ocr"),
    )


def test_first_two_unhealthy_falls_back_to_tertiary(
    patched_builders: dict[str, Any],
) -> None:
    patched_builders["mlx_vlm_healthy"] = False
    patched_builders["ollama_primary_healthy"] = False
    selection = get_ocr_engine(_settings("mlx-vlm"))
    assert selection.provider == "ollama"
    assert selection.model == "qwen2.5-vl:32b"
    assert selection.fell_back is True
    assert selection.attempted == (
        ("mlx-vlm", "mlx-community/DeepSeek-OCR-2-6bit"),
        ("ollama", "deepseek-ocr"),
        ("ollama", "qwen2.5-vl:32b"),
    )


def test_all_links_unhealthy_raises_ocr_error(
    patched_builders: dict[str, Any],
) -> None:
    patched_builders["mlx_vlm_healthy"] = False
    patched_builders["ollama_primary_healthy"] = False
    patched_builders["ollama_fallback_healthy"] = False
    with pytest.raises(OCRError) as excinfo:
        get_ocr_engine(_settings("mlx-vlm"))
    msg = str(excinfo.value)
    assert "No OCR provider available" in msg
    assert "mlx-vlm" in msg
    assert "ollama" in msg
    assert "qwen2.5-vl:32b" in msg


def test_singleton_returns_same_instance(patched_builders: dict[str, Any]) -> None:
    first = get_ocr_engine(_settings("mlx-vlm"))
    second = get_ocr_engine(_settings("mlx-vlm"))
    assert first is second
    assert first.engine is second.engine


def test_ollama_first_chain_skips_mlx_when_primary_healthy(
    patched_builders: dict[str, Any],
) -> None:
    selection = get_ocr_engine(_settings("ollama"))
    assert selection.provider == "ollama"
    assert selection.model == "deepseek-ocr"
    assert selection.fell_back is False
    assert selection.attempted == (("ollama", "deepseek-ocr"),)


def test_win_profile_no_qwen_fallback_skips_tertiary(
    patched_builders: dict[str, Any],
) -> None:
    # Win flagship: ollama primary, no qwen tertiary, mlx-vlm package absent.
    patched_builders["ollama_primary_healthy"] = False
    patched_builders["mlx_vlm_healthy"] = False
    win_settings = _settings("ollama", ocr_ollama_fallback_model=None)
    with pytest.raises(OCRError) as excinfo:
        get_ocr_engine(win_settings)
    msg = str(excinfo.value)
    assert "deepseek-ocr" in msg
    assert "mlx-vlm" in msg
    assert "qwen" not in msg

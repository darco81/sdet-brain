"""MLXVLMOCREngine tests — exercised with monkeypatched mlx-vlm calls."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from sdet_brain.ocr.mlx_vlm_provider import MLXVLMOCREngine
from sdet_brain.ocr.protocol import OCRError, OCRQualityError, OCRResult


class _FakeGenerationResult:
    """Stand-in for mlx-vlm's GenerationResult (text + peak_memory attrs)."""

    def __init__(self, text: str, peak_memory: float = 0.0) -> None:
        self.text = text
        self.peak_memory = peak_memory


@pytest.fixture
def engine() -> MLXVLMOCREngine:
    return MLXVLMOCREngine(
        model_name="test-model",
        default_prompt="Convert the document to markdown.",
        quality_min_chars=10,
    )


def _patch_loaded(monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine) -> None:
    monkeypatch.setattr(
        engine,
        "_load_model",
        lambda: ("model", "processor", "config"),
    )


def test_extract_text_returns_ocr_result(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    _patch_loaded(monkeypatch, engine)
    monkeypatch.setattr(
        engine,
        "_generate",
        lambda *_args, **_kw: _FakeGenerationResult(
            text="This is a real receipt with enough chars.",
            peak_memory=5120.0,
        ),
    )

    result = engine.extract_text(b"\x89PNG fake bytes")

    assert isinstance(result, OCRResult)
    assert result.text == "This is a real receipt with enough chars."
    assert result.model == "test-model"
    assert result.duration_s >= 0
    assert result.peak_memory_gb == 5.0  # 5120 MB / 1024 = 5.0 GB


def test_extract_text_strips_deepseek_tokens_and_dedupes(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    _patch_loaded(monkeypatch, engine)
    raw = (
        "Header line with content\n"
        "Header line with content\n"
        "Header line with content\n"
        "Header line with content\n"
        "<|ref|>noisy<|/ref|><|det|>[[1,2,3,4]]<|/det|> Real data here"
    )
    monkeypatch.setattr(
        engine,
        "_generate",
        lambda *_a, **_kw: _FakeGenerationResult(text=raw),
    )

    result = engine.extract_text(b"png-bytes")

    # Token strip removed the ref/det envelope.
    assert "<|ref|>" not in result.text
    assert "<|/det|>" not in result.text
    # Dedup kept first two identical lines, dropped the rest.
    assert result.text.count("Header line with content") == 2
    assert "Real data here" in result.text


def test_extract_text_raises_quality_error_when_too_short(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    _patch_loaded(monkeypatch, engine)
    monkeypatch.setattr(
        engine,
        "_generate",
        lambda *_a, **_kw: _FakeGenerationResult(text="hi"),
    )

    with pytest.raises(OCRQualityError) as excinfo:
        engine.extract_text(b"png-bytes")
    assert "below quality bar" in str(excinfo.value)
    assert "min=10" in str(excinfo.value)


def test_extract_text_handles_string_output(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    # mlx-vlm 0.5.x sometimes returns a raw str rather than GenerationResult.
    _patch_loaded(monkeypatch, engine)
    monkeypatch.setattr(
        engine,
        "_generate",
        lambda *_a, **_kw: "Plain string output long enough.",
    )

    result = engine.extract_text(b"png-bytes")
    assert result.text == "Plain string output long enough."
    assert result.peak_memory_gb is None


def test_extract_text_rejects_empty_bytes(engine: MLXVLMOCREngine) -> None:
    with pytest.raises(OCRError, match="Empty image_bytes"):
        engine.extract_text(b"")


def test_extract_text_passes_custom_prompt(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    _patch_loaded(monkeypatch, engine)
    captured: dict[str, Any] = {}

    def fake_generate(
        _m: Any, _p: Any, _c: Any, prompt: str, image_path: Path
    ) -> _FakeGenerationResult:
        captured["prompt"] = prompt
        captured["path"] = image_path
        # tmpfile is alive RIGHT NOW (inside the NamedTemporaryFile ctx);
        # after extract_text returns it'll be unlinked. Capture the state here.
        captured["path_exists"] = image_path.exists()
        captured["payload"] = image_path.read_bytes()
        return _FakeGenerationResult(text="Long enough output text here.")

    monkeypatch.setattr(engine, "_generate", fake_generate)

    engine.extract_text(b"png-bytes", prompt="Custom prompt please.")

    assert captured["prompt"] == "Custom prompt please."
    assert isinstance(captured["path"], Path)
    assert captured["path_exists"] is True
    assert captured["payload"] == b"png-bytes"


def test_extract_text_singleton_load_called_once(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    load_calls: list[int] = []

    def fake_load() -> tuple[str, str, str]:
        load_calls.append(1)
        return ("model", "processor", "config")

    monkeypatch.setattr(engine, "_load_model", fake_load)
    monkeypatch.setattr(
        engine,
        "_generate",
        lambda *_a, **_kw: _FakeGenerationResult(text="Long enough text content."),
    )

    engine.extract_text(b"first")
    engine.extract_text(b"second")
    engine.extract_text(b"third")

    assert len(load_calls) == 1


def test_health_check_returns_true_when_mlx_vlm_importable(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    # Inject a fake mlx_vlm module so the test passes regardless of
    # whether the real package is installed (cross-platform: Win CI
    # doesn't have mlx_vlm; we still want this test to assert the
    # contract "health_check returns True when the module exists").
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    assert engine.health_check() is True


def test_health_check_returns_false_when_mlx_vlm_missing(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    # Standard Python trick: assigning None to sys.modules entry forces
    # subsequent `import mlx_vlm` to raise ImportError.
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)
    assert engine.health_check() is False


def test_extract_text_clears_mlx_cache_even_when_generate_raises(
    monkeypatch: pytest.MonkeyPatch, engine: MLXVLMOCREngine
) -> None:
    clear_calls: list[int] = []
    fake_mx_core = types.ModuleType("mlx.core")
    fake_mx_core.clear_cache = lambda: clear_calls.append(1)  # type: ignore[attr-defined]
    fake_mx = types.ModuleType("mlx")
    fake_mx.core = fake_mx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx_core)

    _patch_loaded(monkeypatch, engine)

    def boom(*_args: Any, **_kw: Any) -> Any:
        raise RuntimeError("simulated mlx-vlm failure")

    monkeypatch.setattr(engine, "_generate", boom)

    with pytest.raises(RuntimeError, match="simulated mlx-vlm failure"):
        engine.extract_text(b"png-bytes")

    assert clear_calls == [1]

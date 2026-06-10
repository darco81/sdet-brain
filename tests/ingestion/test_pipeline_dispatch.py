"""Pipeline dispatch helpers: ``maybe_build_ocr_engine`` + fatal-error policy."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from sdet_brain.config import Settings
from sdet_brain.ingestion import pipeline as pipeline_module
from sdet_brain.ingestion.pipeline import maybe_build_ocr_engine
from sdet_brain.ocr import factory as factory_module
from sdet_brain.ocr.factory import reset_ocr_engine
from sdet_brain.ocr.protocol import OCRResult


def _write_real_png(path: Path) -> None:
    Image.new("RGB", (40, 40), color="white").save(path, format="PNG")


class _StubOCREngine:
    def __init__(self, *, model_name: str = "stub") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def extract_text(self, image_bytes: bytes, *, prompt: str | None = None) -> OCRResult:
        _ = image_bytes, prompt
        return OCRResult(text="stub", model=self._model_name, duration_s=0.001)

    def health_check(self) -> bool:
        return True


@pytest.fixture
def patched_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Any]]:
    state: dict[str, Any] = {"build_calls": 0}

    def build_stub(_: Settings, model: str) -> _StubOCREngine:
        state["build_calls"] += 1
        return _StubOCREngine(model_name=model)

    monkeypatch.setitem(factory_module._BUILDERS, "mlx-vlm", build_stub)
    monkeypatch.setitem(factory_module._BUILDERS, "ollama", build_stub)
    reset_ocr_engine()
    yield state
    reset_ocr_engine()


def _settings() -> Settings:
    return Settings()


# --- maybe_build_ocr_engine -----------------------------------------------


def test_maybe_build_returns_none_for_single_markdown_file(
    tmp_path: Path, patched_builders: dict[str, Any]
) -> None:
    md = tmp_path / "note.md"
    md.write_text("# hello", encoding="utf-8")

    result = maybe_build_ocr_engine(md, _settings())

    assert result is None
    assert patched_builders["build_calls"] == 0


def test_maybe_build_returns_engine_for_single_image_file(
    tmp_path: Path, patched_builders: dict[str, Any]
) -> None:
    png = tmp_path / "scan.png"
    png.write_bytes(b"\x89PNG fake")

    result = maybe_build_ocr_engine(png, _settings())

    assert result is not None
    assert patched_builders["build_calls"] == 1


def test_maybe_build_returns_engine_for_single_pdf_file(
    tmp_path: Path, patched_builders: dict[str, Any]
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF fake")

    result = maybe_build_ocr_engine(pdf, _settings())

    assert result is not None
    assert patched_builders["build_calls"] == 1


def test_maybe_build_returns_none_for_markdown_only_dir(
    tmp_path: Path, patched_builders: dict[str, Any]
) -> None:
    (tmp_path / "a.md").write_text("# a", encoding="utf-8")
    (tmp_path / "b.md").write_text("# b", encoding="utf-8")

    result = maybe_build_ocr_engine(tmp_path, _settings())

    assert result is None
    assert patched_builders["build_calls"] == 0


def test_maybe_build_returns_engine_for_mixed_dir(
    tmp_path: Path, patched_builders: dict[str, Any]
) -> None:
    (tmp_path / "note.md").write_text("# x", encoding="utf-8")
    (tmp_path / "receipt.jpg").write_bytes(b"\xff\xd8\xff\xe0 jpg")

    result = maybe_build_ocr_engine(tmp_path, _settings())

    assert result is not None
    assert patched_builders["build_calls"] >= 1


def test_maybe_build_singleton_within_run(tmp_path: Path, patched_builders: dict[str, Any]) -> None:
    """Two calls in a row return the same cached engine (factory singleton)."""
    (tmp_path / "scan.pdf").write_bytes(b"%PDF fake")

    first = maybe_build_ocr_engine(tmp_path, _settings())
    second = maybe_build_ocr_engine(tmp_path, _settings())

    assert first is second
    assert patched_builders["build_calls"] == 1


# --- fatal vs recoverable in ingest_path loop -----------------------------


class _BoomEngine:
    @property
    def model_name(self) -> str:
        return "boom"

    def extract_text(self, *_a: Any, **_kw: Any) -> OCRResult:
        raise MemoryError("simulated MLX OOM mid-ingest")

    def health_check(self) -> bool:
        return True


def test_memory_error_aborts_ingest_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patched_builders: dict[str, Any],
) -> None:
    """A MemoryError mid-loop must propagate, not be silently appended
    to stats.errors as a string."""
    # Stub the heavy collaborators that ingest_path needs.

    class _FakeStorage:
        client = type("C", (), {"scroll": lambda *a, **kw: ([], None)})()

        def delete_by_filter(self, *a: Any, **kw: Any) -> None:
            pass

        def upsert_points(self, *a: Any, **kw: Any) -> None:
            pass

    class _FakeEmbedder:
        vector_size = 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def health_check(self) -> bool:
            return True

    monkeypatch.setattr(
        pipeline_module,
        "get_sparse_embedder",
        lambda: _FakeEmbedder(),
    )

    image = tmp_path / "explode.png"
    _write_real_png(image)

    with pytest.raises(MemoryError, match="simulated"):
        pipeline_module.ingest_path(
            image,
            _FakeStorage(),  # type: ignore[arg-type]
            _FakeEmbedder(),  # type: ignore[arg-type]
            ocr_engine=_BoomEngine(),  # type: ignore[arg-type]
            settings=_settings(),
        )


def test_per_file_ocr_error_appended_to_stats_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Domain errors (OCRError, OCRQualityError) stay per-file —
    they do NOT abort the run."""
    from sdet_brain.ocr.protocol import OCRError

    class _PerFileBoom:
        @property
        def model_name(self) -> str:
            return "boom-per-file"

        def extract_text(self, *_a: Any, **_kw: Any) -> OCRResult:
            raise OCRError("simulated bad image — skip this one")

        def health_check(self) -> bool:
            return True

    class _FakeStorage:
        client = type("C", (), {"scroll": lambda *a, **kw: ([], None)})()

        def delete_by_filter(self, *a: Any, **kw: Any) -> None:
            pass

        def upsert_points(self, *a: Any, **kw: Any) -> None:
            pass

    class _FakeEmbedder:
        vector_size = 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * 8 for _ in texts]

        def health_check(self) -> bool:
            return True

    monkeypatch.setattr(
        pipeline_module,
        "get_sparse_embedder",
        lambda: _FakeEmbedder(),
    )

    image = tmp_path / "bad.png"
    _write_real_png(image)

    stats = pipeline_module.ingest_path(
        image,
        _FakeStorage(),  # type: ignore[arg-type]
        _FakeEmbedder(),  # type: ignore[arg-type]
        ocr_engine=_PerFileBoom(),  # type: ignore[arg-type]
        settings=_settings(),
    )

    assert len(stats.errors) == 1
    src, msg = stats.errors[0]
    assert "bad.png" in src
    assert "simulated bad image" in msg

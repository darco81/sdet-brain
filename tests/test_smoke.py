"""Smoke tests verifying the package imports cleanly."""

from __future__ import annotations

import sdet_brain
from sdet_brain.config import Settings


def test_package_version_exposed() -> None:
    assert sdet_brain.__version__ == "0.1.0"


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.qdrant_url.startswith("http")
    assert settings.collection_name == "sdet_brand_v1"
    assert settings.embedding_provider in {"mlx", "gemini"}
    assert settings.mlx_vector_size == 1024
    assert settings.gemini_vector_size == 1024
    assert 0.0 < settings.chunk_overlap_ratio < 1.0

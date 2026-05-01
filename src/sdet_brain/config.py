"""Application configuration loaded from environment variables.

All settings are read from environment or `.env` file via pydantic-settings.
See `.env.example` for the full list of supported variables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["mlx", "gemini"]


class Settings(BaseSettings):
    """Runtime settings for the SDET Brain server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Qdrant ---
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant HTTP endpoint.",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Optional Qdrant API key (production deploys).",
    )
    collection_name: str = Field(
        default="sdet_brand_v1",
        description="Primary Qdrant collection for the brand corpus.",
    )

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = Field(
        default="mlx",
        description="Primary embedding provider. Falls back to the other on failure.",
    )
    mlx_model: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="HuggingFace model id for MLX local embeddings.",
    )
    mlx_vector_size: int = Field(
        default=1024,
        description="Output dimensionality for the MLX embedding model.",
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key (used for fallback embeddings).",
    )
    gemini_embedding_model: str = Field(
        default="text-embedding-004",
        description="Gemini embedding model id.",
    )
    gemini_vector_size: int = Field(
        default=768,
        description="Output dimensionality for the Gemini embedding model.",
    )

    # --- Server ---
    server_host: str = Field(default="127.0.0.1")
    server_port: int = Field(default=8080)
    mcp_sse_port: int = Field(default=8081)
    log_level: str = Field(default="INFO")

    # --- Ingestion ---
    chunk_target_chars: int = Field(default=800)
    chunk_overlap_ratio: float = Field(default=0.15)
    embed_batch_size: int = Field(default=32)
    watch_paths: str = Field(
        default="",
        description="Comma-separated absolute paths the watcher monitors.",
    )
    watcher_debounce_ms: int = Field(default=300)

    # --- Brand corpus source paths (per source_type) ---
    # Each is a comma-separated list of absolute directories. CLI
    # handlers (ingest, watcher) consume these to wire up the source
    # classifier. Empty means "no roots registered for that
    # source_type" - files outside all roots tag as `unknown`.
    project_knowledge_paths: str = Field(
        default="",
        description="Comma-separated paths whose 01-PROJECT-CONTEXT/etc files map to project-knowledge.",
    )
    drafts_paths: str = Field(
        default="",
        description="Comma-separated paths to draft Markdown trees.",
    )
    articles_paths: str = Field(
        default="",
        description="Comma-separated paths to published article trees.",
    )
    sprint_reports_paths: str = Field(
        default="",
        description="Comma-separated paths to sprint-report directories.",
    )
    brief_paths: str = Field(
        default="",
        description="Comma-separated paths to brief / spec / methodology trees.",
    )

    # --- Reranking (T2-04) ---
    rerank_enabled: bool = Field(
        default=False,
        description="When True, search re-orders candidates with a cross-encoder before returning.",
    )
    rerank_model: str = Field(
        default="jinaai/jina-reranker-v2-base-multilingual",
        description="Cross-encoder model id (must be in fastembed's CROSS_ENCODER_REGISTRY).",
    )
    rerank_top_k_retrieve: int = Field(
        default=30,
        description="How many candidates to over-fetch from Qdrant before reranking.",
    )
    rerank_top_k_return: int = Field(
        default=5,
        description="Top-K to return after reranking.",
    )


def parse_path_list(value: str) -> list[str]:
    """Split a comma-separated env var into a clean list of paths."""
    return [item.strip() for item in value.split(",") if item.strip()]


def get_settings() -> Settings:
    """Return a cached Settings instance.

    The function is intentionally a thin wrapper so callers can monkeypatch
    it in tests without poking at the global module state.
    """
    return Settings()


def project_root() -> Path:
    """Return the repository root path on disk."""
    return Path(__file__).resolve().parents[2]

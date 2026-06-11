"""Application configuration loaded from environment variables.

All settings are read from environment or `.env` file via pydantic-settings.
See `.env.example` for the full list of supported variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["mlx", "gemini"]
OCRProvider = Literal["mlx-vlm", "ollama"]


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
        default="mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
        description="HuggingFace model id for MLX local embeddings.",
    )
    mlx_vector_size: int = Field(
        default=1024,
        description="Output dimensionality after any MRL truncation.",
    )
    mlx_mrl_truncate_to: int | None = Field(
        default=1024,
        description=(
            "Matryoshka truncation length. The 8B Qwen3-Embedding emits "
            "4096 dims natively; setting this to 1024 keeps the leading "
            "slice (~95% retention) so the existing collection schema "
            "stays compatible. Set to None to keep the native dimension."
        ),
    )
    gemini_api_key: str | None = Field(
        default=None,
        description="Google Gemini API key (used for fallback embeddings).",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model id (text-embedding-004 is deprecated).",
    )
    gemini_vector_size: int = Field(
        default=1024,
        description=(
            "Output dimensionality requested from Gemini (via "
            "output_dimensionality). Defaults to 1024 to match the MLX "
            "collection so the cloud fallback stays compatible."
        ),
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

    # --- Local LLM (T2-05) ---
    llm_model: str = Field(
        default="mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        description="Local MLX model id used for instruct-tier tasks (chat, summarize).",
    )
    llm_max_tokens: int = Field(
        default=512,
        description="Default upper bound on generated tokens per LLM call.",
    )
    llm_temperature: float = Field(
        default=0.7,
        description="Default sampling temperature (0.0 = deterministic, 1.0 = creative).",
    )

    # --- LLM routing tiers (T4-03) ---
    llm_routing_enabled: bool = Field(
        default=True,
        description="When False, every task uses ``llm_model`` (v0.3.0 behaviour).",
    )
    llm_fast_model: str = Field(
        default="mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
        description="Fast tier (HyDE rewrites, simple expansions).",
    )
    llm_reasoning_model: str = Field(
        default="mlx-community/Qwen3-Next-80B-A3B-Thinking-4bit",
        description="Reasoning tier (decomposition, judging).",
    )
    llm_router_cache_size: int = Field(
        default=1,
        ge=1,
        description=(
            "Max concurrently-resident MLX LLM models in the router cache. "
            "Each Qwen3-Next-80B-4bit weighs ~40 GB; default of 1 caps "
            "resident weights on a 64 GB Mac. Increase to 2 on hosts with "
            ">=96 GB unified memory for warm fast+instruct."
        ),
    )

    # --- OCR (v0.6.0) ---
    ocr_provider: OCRProvider = Field(
        default="mlx-vlm",
        description=(
            "Primary OCR backend. ``mlx-vlm`` uses local MLX (Mac flagship); "
            "``ollama`` uses an Ollama server (cross-platform). Factory falls "
            "back along the configured chain on health-check failure."
        ),
    )
    ocr_mlx_vlm_model: str = Field(
        default="mlx-community/DeepSeek-OCR-2-6bit",
        description="HuggingFace model id for the MLX-VLM OCR provider.",
    )
    ocr_ollama_primary_model: str = Field(
        default="deepseek-ocr",
        description="Ollama model tag used as the primary OCR model on either platform.",
    )
    ocr_ollama_fallback_model: str | None = Field(
        default="qwen2.5-vl:32b",
        description=(
            "Heavyweight multilingual fallback (~40 s/img on Mac). Set to "
            "``None`` on hosts that cannot fit it (Win 4 GB VRAM)."
        ),
    )
    ocr_timeout_seconds: int = Field(
        default=120,
        ge=1,
        description="Wall-clock cap for a single OCR call before ``OCRTimeoutError``.",
    )
    ocr_max_image_dim: int = Field(
        default=1600,
        ge=64,
        description="Resize budget — long edge clamped to this many pixels before OCR.",
    )
    ocr_max_image_bytes: int = Field(
        default=20_000_000,
        ge=1,
        description="Hard ceiling on input image size; bigger payloads are rejected.",
    )
    ocr_max_pdf_pages: int = Field(
        default=20,
        ge=1,
        description="Hard ceiling on PDF page count; longer documents are rejected.",
    )
    ocr_keep_alive: str = Field(
        default="5m",
        description=(
            "Ollama ``keep_alive`` directive — how long the model stays "
            "loaded after the last request before being unloaded."
        ),
    )
    ocr_quality_min_chars: int = Field(
        default=50,
        ge=0,
        description=(
            "Below this character count (after grounding-token strip) the "
            "provider raises ``OCRQualityError`` and the factory tries the "
            "next link in the fallback chain."
        ),
    )
    ocr_pii_scrub: bool = Field(
        default=False,
        description=(
            "Feature flag for post-OCR PII scrubbing (NIP, account numbers, "
            "personal data). Off in v0.6.0 MVP; hook reserved for v0.6.1."
        ),
    )
    ocr_grounding_prompt: str = Field(
        default="<|grounding|>Convert the document to markdown.",
        description="Prompt fed to DeepSeek-OCR variants — uses the grounding token.",
    )
    ocr_general_prompt: str = Field(
        default=(
            "Extract all text from this image and return it as markdown. "
            "Preserve layout and structure."
        ),
        description="Prompt fed to general VLMs (Qwen2.5-VL) that lack the grounding token.",
    )


def parse_path_list(value: str) -> list[str]:
    """Split a comma-separated env var into a clean list of paths."""
    return [item.strip() for item in value.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    Caching means the ``.env`` / environment is parsed once, not on every
    ``Depends(get_settings)`` call. Call ``get_settings.cache_clear()`` in
    tests that need to observe changed environment between constructions.
    """
    return Settings()


def project_root() -> Path:
    """Return the repository root path on disk."""
    return Path(__file__).resolve().parents[2]

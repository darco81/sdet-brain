"""End-to-end ingestion pipeline.

`ingest_path` walks a file or directory, runs each ``.md`` file through
the parser/chunker/embedder/storage layers, and reports
:class:`IngestStats`. Re-ingestion is idempotent: an unchanged
``content_hash`` short-circuits the file unless ``force_reindex`` is
set.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
)
from qdrant_client.models import (
    SparseVector as QdrantSparseVector,
)

from sdet_brain.embeddings.sparse_embedder import (
    ISparseEmbedder,
    SparseVector,
    get_sparse_embedder,
)
from sdet_brain.ingestion.document_parser import parse_markdown
from sdet_brain.ingestion.frontmatter_schema import to_payload_fields
from sdet_brain.ingestion.image_parser import (
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    is_image_path,
    is_pdf_path,
    parse_image,
    parse_pdf,
)
from sdet_brain.ingestion.models import Chunk, ParsedDocument
from sdet_brain.ingestion.source_classifier import SourceConfig, classify_source
from sdet_brain.storage.collections import (
    COLLECTION_NAME,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    utc_now_iso,
)

if TYPE_CHECKING:
    from sdet_brain.config import Settings
    from sdet_brain.embeddings.protocol import IEmbedder
    from sdet_brain.ocr.protocol import IOCREngine
    from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE: Final[int] = 32
CHUNK_NAMESPACE: Final[uuid.UUID] = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://sdet-brain/chunks"
)
INGESTIBLE_SUFFIXES: Final[frozenset[str]] = (
    frozenset({".md"}) | IMAGE_SUFFIXES | PDF_SUFFIXES
)


@dataclass
class IngestStats:
    """Aggregate counters returned by `ingest_path`."""

    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_replaced: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Processed {self.files_processed} files, "
            f"created {self.chunks_created} chunks, "
            f"skipped {self.files_skipped} files (cache), "
            f"replaced {self.chunks_replaced} chunks"
            + (f", {len(self.errors)} errors" if self.errors else "")
        )


def _iter_ingestible_files(
    root: Path, exclude_dirs: tuple[Path, ...] = ()
) -> Iterator[Path]:
    """Yield ingestible files under ``root`` (or just ``root`` itself).

    Ingestible suffixes (case-insensitive): ``.md`` plus everything in
    :data:`sdet_brain.ingestion.image_parser.IMAGE_SUFFIXES` plus
    :data:`PDF_SUFFIXES` — markdown, images, PDFs.

    Hidden path parts (anything starting with ``.``) are always
    skipped. ``exclude_dirs`` accepts two forms:

    - **Subtree paths** (absolute or relative-with-slashes), e.g.
      ``/abs/v0.4-planning`` or ``./v0.4-planning`` - anything under
      that resolved path is dropped.
    - **Bare directory names** with no slash, e.g. ``node_modules``
      or ``__pycache__`` - every path part with that name matches,
      regardless of where it appears in the tree (gitignore-style).
    """
    name_excludes: set[str] = set()
    abs_excludes: list[Path] = []
    for d in exclude_dirs:
        if not d.is_absolute() and len(d.parts) == 1:
            name_excludes.add(d.name)
        else:
            abs_excludes.append(d.resolve())

    def _is_excluded(path: Path) -> bool:
        if any(part in name_excludes for part in path.parts):
            return True
        resolved = path.resolve()
        for parent in abs_excludes:
            try:
                resolved.relative_to(parent)
                return True
            except ValueError:
                continue
        return False

    if root.is_file():
        if root.suffix.lower() in INGESTIBLE_SUFFIXES and not _is_excluded(root):
            yield root
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in INGESTIBLE_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if _is_excluded(path):
            continue
        yield path


def _chunk_point_id(source_path: str, chunk_index: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{source_path}#{chunk_index}"))


def _existing_hash(
    storage: QdrantStorage, collection: str, source_path: str
) -> str | None:
    """Return the ``content_hash`` of any chunk currently stored for ``source_path``."""
    points, _ = storage.client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return None
    payload = points[0].payload or {}
    value = payload.get("content_hash")
    return value if isinstance(value, str) else None


def _load_existing_hashes(
    storage: QdrantStorage, collection: str, source_paths: list[str]
) -> dict[str, str]:
    """Return ``{source_path: content_hash}`` for every path with chunks.

    Single Qdrant scroll using ``MatchAny`` on the union of paths -
    O(1) round-trips instead of one per file. The page size is
    ``len(paths) * 2`` so we read at least one chunk per source plus
    a buffer; we do NOT need every chunk because all chunks of a file
    share the same `content_hash`.
    """
    if not source_paths:
        return {}
    page_size = max(64, min(len(source_paths) * 2, 1024))
    seen: dict[str, str] = {}
    offset: object = None
    while True:
        page, offset = storage.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_path",
                        match=MatchAny(any=source_paths),
                    )
                ]
            ),
            limit=page_size,
            offset=offset,  # type: ignore[arg-type]
            with_payload=True,
            with_vectors=False,
        )
        for point in page:
            payload = point.payload or {}
            path = payload.get("source_path")
            content_hash = payload.get("content_hash")
            if (
                isinstance(path, str)
                and isinstance(content_hash, str)
                and path not in seen
            ):
                seen[path] = content_hash
        # Early exit once we have a hash for every path we asked about.
        if len(seen) == len(source_paths):
            return seen
        if offset is None:
            return seen


def _delete_existing_chunks(
    storage: QdrantStorage, collection: str, source_path: str
) -> None:
    storage.delete_by_filter(
        collection,
        Filter(
            must=[FieldCondition(key="source_path", match=MatchValue(value=source_path))]
        ),
    )


def _build_payload(
    document: ParsedDocument,
    chunk: Chunk,
    source_type: str,
    created_at: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": chunk.text,
        "source_path": document.source_path,
        "source_type": source_type,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "heading_path": chunk.heading_path,
        "has_code": chunk.has_code,
        "char_count": chunk.char_count,
        "token_estimate": chunk.token_estimate,
        "frontmatter": dict(document.frontmatter),
        "content_hash": document.content_hash,
        "created_at": created_at,
    }
    if document.brand_frontmatter is not None:
        # Lift validated fields to top-level keys so Qdrant can build
        # payload indexes on them (faster than nested-key filtering).
        payload.update(to_payload_fields(document.brand_frontmatter))
    return payload


def _embed_in_batches(
    embedder: IEmbedder,
    chunks: tuple[Chunk, ...],
    batch_size: int,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(embedder.embed([chunk.text for chunk in batch]))
    return vectors


def _sparse_in_batches(
    sparse_embedder: ISparseEmbedder,
    chunks: tuple[Chunk, ...],
    batch_size: int,
) -> list[SparseVector]:
    """BM25-encode chunks in the same batch shape as the dense embedder."""
    vectors: list[SparseVector] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(sparse_embedder.embed([chunk.text for chunk in batch]))
    return vectors


def _build_named_vector(dense: list[float], sparse: SparseVector) -> Any:
    """Pack dense + sparse into the dict shape Qdrant expects for upsert.

    The return type is ``Any`` because the qdrant-client `PointStruct`
    declares its ``vector`` field with an invariant ``dict`` whose
    value union we'd otherwise have to spell out exhaustively. A
    runtime-correct dict is enough; the type-system gymnastics around
    that union belong inside the SDK, not us.
    """
    return {
        DENSE_VECTOR_NAME: dense,
        SPARSE_VECTOR_NAME: QdrantSparseVector(
            indices=sparse.indices, values=sparse.values
        ),
    }


def _ingest_document(
    document: ParsedDocument,
    storage: QdrantStorage,
    embedder: IEmbedder,
    sparse_embedder: ISparseEmbedder,
    collection: str,
    source_type: str,
    batch_size: int,
    *,
    force_reindex: bool,
    stats: IngestStats,
    cached_hash: str | None = None,
    cached_hash_known: bool = False,
) -> None:
    if not document.chunks:
        stats.files_skipped += 1
        return

    if cached_hash_known:
        existing = cached_hash
    else:
        existing = _existing_hash(storage, collection, document.source_path)
    if not force_reindex and existing == document.content_hash:
        stats.files_skipped += 1
        return

    if existing is not None:
        _delete_existing_chunks(storage, collection, document.source_path)
        stats.chunks_replaced += 1

    vectors = _embed_in_batches(embedder, document.chunks, batch_size)
    if len(vectors) != len(document.chunks):
        raise RuntimeError(
            f"Embedder returned {len(vectors)} vectors for "
            f"{len(document.chunks)} chunks - aborting upsert."
        )
    sparse_vectors = _sparse_in_batches(sparse_embedder, document.chunks, batch_size)
    if len(sparse_vectors) != len(document.chunks):
        raise RuntimeError(
            f"Sparse embedder returned {len(sparse_vectors)} vectors for "
            f"{len(document.chunks)} chunks - aborting upsert."
        )

    created_at = utc_now_iso()
    points = [
        PointStruct(
            id=_chunk_point_id(document.source_path, chunk.chunk_index),
            vector=_build_named_vector(vectors[idx], sparse_vectors[idx]),
            payload=_build_payload(document, chunk, source_type, created_at),
        )
        for idx, chunk in enumerate(document.chunks)
    ]
    storage.upsert_points(collection, points)
    stats.chunks_created += len(points)
    stats.files_processed += 1


def maybe_build_ocr_engine(
    target: Path, settings: Settings
) -> IOCREngine | None:
    """Build an OCR engine if ``target`` will need one, else return ``None``.

    Pre-scans the target so markdown-only paths skip OCR backend
    bootstrap entirely. The first image/PDF encountered triggers
    :func:`sdet_brain.ocr.factory.get_ocr_engine` which honours the
    process-wide singleton, so subsequent ingests share weights.
    """
    from sdet_brain.ocr.factory import get_ocr_engine

    def _needs_ocr(path: Path) -> bool:
        return is_image_path(path) or is_pdf_path(path)

    if target.is_file():
        if _needs_ocr(target):
            return get_ocr_engine(settings).engine
        return None

    for candidate in _iter_ingestible_files(target):
        if _needs_ocr(candidate):
            return get_ocr_engine(settings).engine
    return None


def _parse_one(
    path: Path,
    *,
    ocr_engine: IOCREngine | None,
    settings: Settings | None,
) -> ParsedDocument:
    """Dispatch to the right parser based on filesystem suffix."""
    if is_image_path(path) or is_pdf_path(path):
        if ocr_engine is None or settings is None:
            raise RuntimeError(
                f"OCR engine + settings required to ingest {path.suffix} files; "
                "pass `ocr_engine=` and `settings=` to ``ingest_path``.",
            )
        if is_pdf_path(path):
            return parse_pdf(path, ocr_engine=ocr_engine, settings=settings)
        return parse_image(path, ocr_engine=ocr_engine, settings=settings)
    return parse_markdown(path)


def ingest_path(
    path: Path,
    storage: QdrantStorage,
    embedder: IEmbedder,
    *,
    source_config: SourceConfig | None = None,
    collection: str = COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force_reindex: bool = False,
    exclude_dirs: tuple[Path, ...] = (),
    progress: Iterator[Path] | None = None,
    sparse_embedder: ISparseEmbedder | None = None,
    ocr_engine: IOCREngine | None = None,
    settings: Settings | None = None,
) -> IngestStats:
    """Walk ``path`` and ingest every ingestible file beneath it.

    Ingestible suffixes: ``.md`` (markdown), ``.jpg``/``.png``/``.heic``
    /... (image OCR), ``.pdf`` (PDF page OCR). Pass ``ocr_engine`` +
    ``settings`` to enable image/PDF ingestion; without them, image and
    PDF files surface as per-file errors in :class:`IngestStats`.

    Parameters
    ----------
    path:
        File or directory.
    storage:
        Configured `QdrantStorage`.
    embedder:
        Provider compatible with `IEmbedder`.
    source_config:
        Optional path heuristics for `classify_source`. Files outside
        all registered roots are tagged ``unknown``.
    collection:
        Target collection name.
    batch_size:
        Number of chunks per embedding call.
    force_reindex:
        When True, ignore the cached `content_hash` and re-embed every
        file regardless.
    progress:
        Optional iterable wrapper (e.g. `tqdm`) used for visible
        progress reporting. The CLI passes `tqdm` here; tests pass
        ``None``.
    ocr_engine:
        Configured OCR engine. Required if ``path`` contains any image
        or PDF file; ignored when only markdown is encountered.
    settings:
        :class:`Settings` instance carrying ``ocr_*`` budgets (image
        size cap, PDF page cap, resize target). Required alongside
        ``ocr_engine``.
    """
    config = source_config or SourceConfig()
    sparse = sparse_embedder if sparse_embedder is not None else get_sparse_embedder()
    files = list(_iter_ingestible_files(path, exclude_dirs=exclude_dirs))
    iterator: Iterator[Path] = progress if progress is not None else iter(files)

    # Single batched cache-check: load every existing content_hash for
    # the files we're about to walk in one Qdrant scroll. O(1)
    # round-trips instead of O(N). For single-file ingests we skip the
    # batch and let the per-file fallback handle it.
    cached_hashes: dict[str, str] = {}
    if len(files) > 1:
        cached_hashes = _load_existing_hashes(
            storage, collection, [str(f) for f in files]
        )

    stats = IngestStats()
    for file_path in iterator:
        try:
            document = _parse_one(
                file_path, ocr_engine=ocr_engine, settings=settings,
            )
            # OCR documents take a distinct source_type so search can filter
            # them out / target them specifically.
            if is_image_path(file_path) or is_pdf_path(file_path):
                source_type = "image-ocr"
            else:
                source_type = classify_source(file_path, config)
            cached = cached_hashes.get(str(file_path))
            _ingest_document(
                document,
                storage,
                embedder,
                sparse,
                collection,
                source_type,
                batch_size,
                force_reindex=force_reindex,
                stats=stats,
                cached_hash=cached,
                cached_hash_known=len(files) > 1,
            )
        except Exception as exc:
            # Per-file failures must not abort the whole walk - record
            # the error and keep going so a single malformed file does
            # not freeze a long ingest.
            logger.exception("Failed to ingest %s", file_path)
            stats.errors.append((str(file_path), str(exc)))
    return stats

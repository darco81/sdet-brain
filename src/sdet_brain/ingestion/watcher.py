"""Filesystem watcher that auto-reindexes Markdown files.

The watcher uses ``watchdog`` to receive change notifications and
keeps the actual ingest work on a worker thread so the observer
thread never blocks. A short debounce window collapses the burst of
events that editors emit on a single save (VS Code, Vim, etc.) into
a single re-ingest.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from qdrant_client.models import FieldCondition, Filter, MatchValue
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from sdet_brain.embeddings.protocol import IEmbedder
from sdet_brain.ingestion.pipeline import IngestStats, ingest_path
from sdet_brain.ingestion.source_classifier import SourceConfig
from sdet_brain.storage.collections import COLLECTION_NAME
from sdet_brain.storage.qdrant_client import QdrantStorage

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_MS: Final[int] = 300
SLEEP_INTERVAL_S: Final[float] = 0.05
IGNORED_PATH_PARTS: Final[frozenset[str]] = frozenset(
    {"node_modules", "__pycache__", ".git", ".venv", "venv"}
)


@dataclass
class WatcherStats:
    """Counters surfaced for tests and the CLI."""

    events_received: int = 0
    events_filtered: int = 0
    files_ingested: int = 0
    files_deleted: int = 0
    chunks_created: int = 0
    chunks_replaced: int = 0


def is_relevant_path(path: Path) -> bool:
    """Return True for `.md` files that are not hidden / vendored."""
    if path.suffix.lower() != ".md":
        return False
    parts = path.parts
    if any(part.startswith(".") and part not in {".", ".."} for part in parts):
        return False
    return not any(part in IGNORED_PATH_PARTS for part in parts)


class BrainWatcher(FileSystemEventHandler):
    """Debounced filesystem watcher that wraps the ingest pipeline."""

    def __init__(
        self,
        watch_paths: Iterable[Path],
        storage: QdrantStorage,
        embedder: IEmbedder,
        *,
        source_config: SourceConfig | None = None,
        collection: str = COLLECTION_NAME,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        sleep_interval: float = SLEEP_INTERVAL_S,
    ) -> None:
        super().__init__()
        self._watch_paths = [Path(p).resolve() for p in watch_paths]
        self._storage = storage
        self._embedder = embedder
        self._source_config = source_config or SourceConfig()
        self._collection = collection
        self._debounce_s = debounce_ms / 1000.0
        self._sleep_interval = sleep_interval

        self._pending: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._observer: BaseObserver | None = None
        self._worker: threading.Thread | None = None
        self.stats = WatcherStats()

    @property
    def watch_paths(self) -> tuple[Path, ...]:
        return tuple(self._watch_paths)

    def start(self) -> None:
        if self._observer is not None:
            return
        observer: BaseObserver = Observer()
        for path in self._watch_paths:
            if not path.exists():
                logger.warning("Watch path does not exist: %s", path)
                continue
            observer.schedule(self, str(path), recursive=True)
            logger.info("Watching %s", path)
        observer.start()
        self._observer = observer
        self._worker = threading.Thread(
            target=self._worker_loop, name="sdet-brain-watcher", daemon=True
        )
        self._worker.start()

    def stop(self, *, drain: bool = True) -> None:
        self._stop_event.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        if drain:
            self._drain_pending()
        logger.info("Watcher shutdown complete (%s)", self.stats)

    def __enter__(self) -> BrainWatcher:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # watchdog hooks
    # ------------------------------------------------------------------

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Treat move as delete-old + ingest-new.
        src = self._coerce_path(getattr(event, "src_path", ""))
        dest = self._coerce_path(getattr(event, "dest_path", ""))
        if src is not None:
            self._delete_now(src)
        if dest is not None and is_relevant_path(dest):
            with self._lock:
                self._pending[dest] = time.monotonic()

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = self._coerce_path(event.src_path)
        if path is None or not is_relevant_path(path):
            self.stats.events_filtered += 1
            return
        self.stats.events_received += 1
        self._delete_now(path)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _enqueue(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = self._coerce_path(event.src_path)
        self.stats.events_received += 1
        if path is None or not is_relevant_path(path):
            self.stats.events_filtered += 1
            return
        with self._lock:
            self._pending[path] = time.monotonic()

    def _coerce_path(self, raw: str | bytes) -> Path | None:
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return Path(raw).resolve()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._process_due()
            self._stop_event.wait(self._sleep_interval)

    def _process_due(self) -> None:
        now = time.monotonic()
        ready: list[Path] = []
        with self._lock:
            for path, last_seen in list(self._pending.items()):
                if now - last_seen >= self._debounce_s:
                    ready.append(path)
                    del self._pending[path]
        for path in ready:
            self._ingest_one(path)

    def _drain_pending(self) -> None:
        with self._lock:
            ready = list(self._pending)
            self._pending.clear()
        for path in ready:
            self._ingest_one(path)

    def _ingest_one(self, path: Path) -> None:
        if not path.exists():
            self._delete_now(path)
            return
        try:
            stats = ingest_path(
                path,
                self._storage,
                self._embedder,
                source_config=self._source_config,
                collection=self._collection,
            )
        except Exception:
            logger.exception("Watcher ingest failed for %s", path)
            return
        self._merge_stats(stats)
        self._log_ingest(path, stats)

    def _delete_now(self, path: Path) -> None:
        try:
            self._storage.delete_by_filter(
                self._collection,
                Filter(
                    must=[
                        FieldCondition(
                            key="source_path",
                            match=MatchValue(value=str(path)),
                        )
                    ]
                ),
            )
        except Exception:
            logger.exception("Watcher delete failed for %s", path)
            return
        self.stats.files_deleted += 1
        logger.info("Removed chunks for deleted file %s", path)

    def _merge_stats(self, stats: IngestStats) -> None:
        self.stats.files_ingested += stats.files_processed
        self.stats.chunks_created += stats.chunks_created
        self.stats.chunks_replaced += stats.chunks_replaced

    def _log_ingest(self, path: Path, stats: IngestStats) -> None:
        if stats.files_skipped:
            logger.debug("Watcher cache hit for %s", path)
            return
        logger.info(
            "Re-ingested %s (%d chunks created, %d replaced)",
            path,
            stats.chunks_created,
            stats.chunks_replaced,
        )

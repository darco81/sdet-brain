"""Shared fixtures for the storage integration tests.

These tests assume a running Qdrant instance reachable at
``QDRANT_URL`` (default ``http://localhost:6333``). They are skipped with
a clear reason when the endpoint is unreachable - we never substitute a
mock per project decision (`SDET-BRAIN-BOOTSTRAP-PROMPT.md`).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest

from sdet_brain.storage.qdrant_client import QdrantStorage

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable(url: str) -> bool:
    try:
        response = httpx.get(f"{url}/readyz", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    if not _qdrant_reachable(QDRANT_URL):
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL} (start docker compose first).")
    return QDRANT_URL


@pytest.fixture
def storage(qdrant_url: str) -> Iterator[QdrantStorage]:
    with QdrantStorage(qdrant_url) as client:
        yield client


@pytest.fixture
def temp_collection(storage: QdrantStorage) -> Iterator[str]:
    """Create a unique throwaway collection and tear it down afterwards."""
    name = f"sdet_brain_test_{os.getpid()}_{id(storage)}"
    yield name
    if storage.collection_exists(name):
        storage.client.delete_collection(collection_name=name)

"""Input-validation coverage for the ``ingest_image`` MCP tool.

End-to-end ingest-and-search coverage lives in ``tests/server/test_tools.py``
(behind the Qdrant-reachable skip). These tests pin the user-visible
errors so Claude gets clean diagnostics when the path is wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sdet_brain.config import Settings
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.tools._helpers import ToolError
from sdet_brain.server.tools.ingest_image import ingest_image


def _state() -> AppState:
    return AppState(settings=Settings(), storage=None, selection=None)


def test_ingest_image_rejects_missing_path() -> None:
    with pytest.raises(ToolError, match="does not exist"):
        ingest_image(_state(), path="/var/sdet-brain-fixtures/does-not-exist.png")


def test_ingest_image_rejects_markdown_file(tmp_path: Path) -> None:
    md = tmp_path / "note.md"
    md.write_text("# heading", encoding="utf-8")

    with pytest.raises(ToolError, match="not an image or PDF"):
        ingest_image(_state(), path=str(md))


def test_ingest_image_rejects_dir_with_no_images(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("nope", encoding="utf-8")

    with pytest.raises(ToolError, match="nothing to OCR"):
        ingest_image(_state(), path=str(tmp_path))

"""Unit tests for the search_cli and main_cli dispatcher.

These tests use mocks — no live Qdrant or embedder required.
"""

from __future__ import annotations

import json
from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from sdet_brain.cli.main_cli import _looks_like_path, main as dispatcher_main
from sdet_brain.cli.search_cli import (
    _emit_json,
    _emit_text,
    _extract_slug,
    _extract_topic,
)


# ---------------------------------------------------------------------------
# _extract_slug
# ---------------------------------------------------------------------------


class TestExtractSlug:
    def test_councils_unix_path(self) -> None:
        p = "/home/user/umysl-pieciu/councils/s1-1-publish-decision/verdict.md"
        assert _extract_slug(p) == "s1-1-publish-decision"

    def test_councils_nested(self) -> None:
        p = "/some/path/councils/my-council-slug/transcript.md"
        assert _extract_slug(p) == "my-council-slug"

    def test_fallback_stem(self) -> None:
        assert _extract_slug("/no/match/here.md") == "here"

    def test_empty_path(self) -> None:
        # Should not raise, return 'unknown' or some string
        result = _extract_slug("")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _extract_topic
# ---------------------------------------------------------------------------


class TestExtractTopic:
    def test_first_h1(self) -> None:
        text = "# My Council Topic\nSome content here."
        assert _extract_topic(text, "/path/to/file.md") == "My Council Topic"

    def test_no_h1_falls_back_to_slug(self) -> None:
        text = "Some text without heading"
        result = _extract_topic(text, "/councils/my-council-slug/verdict.md")
        assert "My Council Slug" in result or "my" in result.lower()

    def test_h1_with_whitespace(self) -> None:
        text = "#   Spaced Title  \nBody"
        assert _extract_topic(text, "/path.md") == "Spaced Title"


# ---------------------------------------------------------------------------
# _emit_json
# ---------------------------------------------------------------------------


class TestEmitJson:
    def _make_point(self, score: float, source_path: str, text: str) -> MagicMock:
        p = MagicMock()
        p.score = score
        p.payload = {"source_path": source_path, "text": text}
        return p

    def test_json_structure(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = Namespace(query="test", source_type="councils", format="json")
        point = self._make_point(
            0.75,
            "/home/user/councils/my-slug/verdict.md",
            "# My Topic\nSome snippet content here.",
        )
        _emit_json(args, [point])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "results" in data
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["slug"] == "my-slug"
        assert r["topic"] == "My Topic"
        assert r["score"] == pytest.approx(0.75, abs=1e-4)
        assert "source_path" not in r  # 'path' is the key
        assert r["path"] == "/home/user/councils/my-slug/verdict.md"
        assert len(r["snippet"]) <= 500

    def test_empty_points(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = Namespace(query="test", source_type=None, format="json")
        _emit_json(args, [])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["results"] == []

    def test_snippet_capped_at_500(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = Namespace(query="test", source_type=None, format="json")
        long_text = "x" * 1000
        point = self._make_point(0.5, "/councils/s/v.md", long_text)
        _emit_json(args, [point])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["results"][0]["snippet"]) == 500


# ---------------------------------------------------------------------------
# _emit_text
# ---------------------------------------------------------------------------


class TestEmitText:
    def _make_point(self, score: float, source_path: str, text: str) -> MagicMock:
        p = MagicMock()
        p.score = score
        p.payload = {"source_path": source_path, "text": text}
        return p

    def test_text_output_contains_slug(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = Namespace(query="q", source_type=None, format="text")
        point = self._make_point(0.8, "/councils/some-slug/v.md", "# Topic\ntext here")
        _emit_text(args, [point])
        captured = capsys.readouterr()
        assert "some-slug" in captured.out
        assert "0.8000" in captured.out

    def test_no_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = Namespace(query="nothing", source_type=None, format="text")
        _emit_text(args, [])
        captured = capsys.readouterr()
        assert "No results" in captured.out


# ---------------------------------------------------------------------------
# dispatcher main_cli
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_looks_like_path_absolute(self) -> None:
        assert _looks_like_path("/home/user/docs")

    def test_looks_like_path_relative(self) -> None:
        assert _looks_like_path("./docs")

    def test_looks_like_path_parent(self) -> None:
        assert _looks_like_path("../docs")

    def test_not_a_path(self) -> None:
        assert not _looks_like_path("search")

    def test_dispatch_search_routes_to_search_main(self) -> None:
        with patch("sdet_brain.cli.search_cli.main", return_value=0) as mock_search:
            result = dispatcher_main(["search", "--query", "test"])
        mock_search.assert_called_once_with(["--query", "test"])
        assert result == 0

    def test_dispatch_ingest_routes_to_ingest_main(self) -> None:
        with patch("sdet_brain.cli.ingest_cli.main", return_value=0) as mock_ingest:
            result = dispatcher_main(["ingest", "/tmp/docs"])
        mock_ingest.assert_called_once_with(["/tmp/docs"])
        assert result == 0

    def test_dispatch_unknown_returns_1(self) -> None:
        result = dispatcher_main(["unknowncmd"])
        assert result == 1

    def test_help_returns_0(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = dispatcher_main(["-h"])
        assert result == 0
        captured = capsys.readouterr()
        assert "search" in captured.out

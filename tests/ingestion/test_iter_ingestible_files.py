"""Unit tests for `_iter_markdown_files` exclude semantics.

The CLI passes `--exclude DIR` as `Path(arg)` (argparse `type=Path`),
so a bare name like `node_modules` becomes a relative path that
`resolve()` anchors to the current working directory. The original
implementation only matched absolute resolved paths, which silently
failed for the bare-name case and let `node_modules` subtrees through.
This test pins down the gitignore-style fix.
"""

from __future__ import annotations

from pathlib import Path

from sdet_brain.ingestion.pipeline import _iter_markdown_files


def _touch(path: Path, text: str = "# stub\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_skips_hidden_directories(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / ".git" / "HEAD.md")
    _touch(tmp_path / ".venv" / "lib" / "x.md")

    found = sorted(p.name for p in _iter_markdown_files(tmp_path))

    assert found == ["kept.md"]


def test_bare_name_exclude_matches_at_any_depth(tmp_path: Path) -> None:
    """Regression: `--exclude node_modules` must drop nested copies.

    Before the fix, `Path("node_modules").resolve()` anchored to CWD,
    so anything under `<repo>/sub/node_modules/...` slipped through and
    polluted the corpus with vendored READMEs.
    """
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / "sub" / "kept.md")
    _touch(tmp_path / "node_modules" / "pkg" / "README.md")
    _touch(tmp_path / "sub" / "node_modules" / "pkg" / "README.md")
    _touch(tmp_path / "deep" / "a" / "b" / "node_modules" / "x.md")

    found = sorted(
        str(p.relative_to(tmp_path))
        for p in _iter_markdown_files(tmp_path, exclude_dirs=(Path("node_modules"),))
    )

    assert found == ["kept.md", "sub/kept.md"]


def test_absolute_exclude_drops_only_matching_subtree(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / "drop" / "x.md")
    _touch(tmp_path / "deep" / "drop" / "y.md")  # different subtree

    found = sorted(
        str(p.relative_to(tmp_path))
        for p in _iter_markdown_files(
            tmp_path, exclude_dirs=(tmp_path / "drop",)
        )
    )

    assert found == ["deep/drop/y.md", "kept.md"]


def test_mixing_bare_names_and_absolute_paths(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / "node_modules" / "x.md")
    _touch(tmp_path / "wip" / "y.md")
    _touch(tmp_path / "nested" / "node_modules" / "z.md")

    found = sorted(
        str(p.relative_to(tmp_path))
        for p in _iter_markdown_files(
            tmp_path,
            exclude_dirs=(Path("node_modules"), tmp_path / "wip"),
        )
    )

    assert found == ["kept.md"]


def test_root_is_single_file(tmp_path: Path) -> None:
    f = tmp_path / "single.md"
    _touch(f)

    found = list(_iter_markdown_files(f))

    assert found == [f]

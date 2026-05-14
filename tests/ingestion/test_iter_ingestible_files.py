"""Unit tests for `_iter_ingestible_files` exclude semantics.

The CLI passes `--exclude DIR` as `Path(arg)` (argparse `type=Path`),
so a bare name like `node_modules` becomes a relative path that
`resolve()` anchors to the current working directory. The original
implementation only matched absolute resolved paths, which silently
failed for the bare-name case and let `node_modules` subtrees through.
This test pins down the gitignore-style fix.
"""

from __future__ import annotations

from pathlib import Path

from sdet_brain.ingestion.pipeline import _iter_ingestible_files


def _touch(path: Path, text: str = "# stub\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_skips_hidden_directories(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / ".git" / "HEAD.md")
    _touch(tmp_path / ".venv" / "lib" / "x.md")

    found = sorted(p.name for p in _iter_ingestible_files(tmp_path))

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
        p.relative_to(tmp_path).as_posix()
        for p in _iter_ingestible_files(tmp_path, exclude_dirs=(Path("node_modules"),))
    )

    assert found == ["kept.md", "sub/kept.md"]


def test_absolute_exclude_drops_only_matching_subtree(tmp_path: Path) -> None:
    _touch(tmp_path / "kept.md")
    _touch(tmp_path / "drop" / "x.md")
    _touch(tmp_path / "deep" / "drop" / "y.md")  # different subtree

    found = sorted(
        p.relative_to(tmp_path).as_posix()
        for p in _iter_ingestible_files(
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
        p.relative_to(tmp_path).as_posix()
        for p in _iter_ingestible_files(
            tmp_path,
            exclude_dirs=(Path("node_modules"), tmp_path / "wip"),
        )
    )

    assert found == ["kept.md"]


def test_root_is_single_file(tmp_path: Path) -> None:
    f = tmp_path / "single.md"
    _touch(f)

    found = list(_iter_ingestible_files(f))

    assert found == [f]


# --- v0.6.0: ingestible suffix expansion (markdown + image + PDF) ----------


def test_yields_image_and_pdf_files_alongside_markdown(tmp_path: Path) -> None:
    _touch(tmp_path / "doc.md")
    (tmp_path / "receipt.jpg").write_bytes(b"\xff\xd8\xff\xe0 fake jpg")
    (tmp_path / "scan.PDF").write_bytes(b"%PDF-1.4 fake")
    (tmp_path / "photo.heic").write_bytes(b"ftyp fake heic")
    (tmp_path / "ignored.txt").write_text("nope")
    (tmp_path / "ignored.docx").write_bytes(b"PK fake docx")

    found = sorted(
        p.name for p in _iter_ingestible_files(tmp_path)
    )

    assert found == ["doc.md", "photo.heic", "receipt.jpg", "scan.PDF"]


def test_single_image_root_yields_itself(tmp_path: Path) -> None:
    image = tmp_path / "lone.png"
    image.write_bytes(b"\x89PNG fake")

    found = list(_iter_ingestible_files(image))

    assert found == [image]


def test_single_pdf_root_yields_itself(tmp_path: Path) -> None:
    pdf = tmp_path / "lone.pdf"
    pdf.write_bytes(b"%PDF fake")

    found = list(_iter_ingestible_files(pdf))

    assert found == [pdf]


def test_unknown_suffix_single_root_returns_empty(tmp_path: Path) -> None:
    other = tmp_path / "lone.txt"
    other.write_text("nope")

    found = list(_iter_ingestible_files(other))

    assert found == []

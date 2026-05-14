"""Real-world smoke test for v0.6.0 OCR — runs against actual Downloads files.

Picks a curated set of real paragons / invoices / iPhone photos from
~/Downloads, runs each through the full OCR factory, and prints text
previews + timing.

Run from repo root:

    uv run python scripts/smoke_v060_ocr_real.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from sdet_brain.config import Settings
from sdet_brain.ingestion.image_parser import (
    is_image_path,
    is_pdf_path,
    parse_image,
    parse_pdf,
)
from sdet_brain.ocr.factory import get_ocr_engine, reset_ocr_engine

DOWNLOADS = Path("/Users/dariusz/Downloads")

# Curated picks — mix of formats and sizes so the factory + image_parser
# exercise the EXIF / HEIC / JPEG / PDF code paths.
CANDIDATES = [
    DOWNLOADS / "IMG_0133.HEIC",   # iPhone HEIC #1
    DOWNLOADS / "IMG_0143.HEIC",   # iPhone HEIC #2
    DOWNLOADS / "IMG_0285.HEIC",   # iPhone HEIC #3 (landscape)
    DOWNLOADS / "IMG_0335.HEIC",   # iPhone HEIC #4
    DOWNLOADS / "IMG_0982.jpeg",   # JPEG (already-known: code screenshot)
    DOWNLOADS / "pro_forma_1_2026_ProForma_16-04-2026.pdf",  # PL invoice PDF
    DOWNLOADS / "PA_3841_2026.pdf",  # another PL invoice PDF
]


def _preview(text: str, *, max_lines: int = 12, max_chars: int = 800) -> str:
    snippet = text[:max_chars]
    lines = snippet.split("\n")[:max_lines]
    return "\n".join(lines)


def _run_one(path: Path, *, engine, settings: Settings) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"\n=== {path.name} ({size_mb:.2f} MB) ===")

    t0 = time.time()
    try:
        if is_pdf_path(path):
            doc = parse_pdf(path, ocr_engine=engine, settings=settings)
        elif is_image_path(path):
            doc = parse_image(path, ocr_engine=engine, settings=settings)
        else:
            return False, f"unsupported suffix: {path.suffix}"
    except Exception as exc:  # noqa: BLE001 — smoke harness, want full error
        return False, f"FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}"

    elapsed = time.time() - t0
    full_text = "\n".join(c.text for c in doc.chunks)

    print(f"  duration:    {elapsed:.2f}s")
    print(f"  chunks:      {len(doc.chunks)}")
    print(f"  content_hash: {doc.content_hash[:12]}…")
    print(f"  ocr_model:   {doc.frontmatter.get('ocr_model')}")
    print(f"  ocr_duration_s: {doc.frontmatter.get('ocr_duration_s')}")
    if "total_pages" in doc.frontmatter:
        print(f"  pages:       {doc.frontmatter['total_pages']}")
    print(f"  text length: {len(full_text)} chars")
    print("  --- preview ---")
    for line in _preview(full_text).split("\n"):
        print(f"  | {line}")
    print("  --- /preview ---")

    return True, f"OK ({elapsed:.1f}s, {len(full_text)} chars)"


def main() -> int:
    print("== v0.6.0 OCR real-world smoke ==\n")
    reset_ocr_engine()
    settings = Settings()

    print(f"Provider chain head: {settings.ocr_provider}")
    print(f"MLX model:           {settings.ocr_mlx_vlm_model}")
    print()

    print("Booting OCR engine...")
    t0 = time.time()
    selection = get_ocr_engine(settings)
    print(f"  provider={selection.provider} model={selection.model} "
          f"fell_back={selection.fell_back} (boot={time.time() - t0:.2f}s)")

    results: list[tuple[Path, bool, str]] = []
    for path in CANDIDATES:
        ok, msg = _run_one(path, engine=selection.engine, settings=settings)
        results.append((path, ok, msg))

    print("\n=== Summary ===")
    for path, ok, msg in results:
        flag = "✓" if ok else "✗"
        print(f"  {flag} {path.name}: {msg}")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_ok}/{len(results)} files OCR'd successfully.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

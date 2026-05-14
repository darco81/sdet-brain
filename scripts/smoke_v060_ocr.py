"""Live smoke test for v0.6.0 OCR — synthetic receipt → factory → result.

Renders a synthetic Polish-style receipt PNG via PIL, runs it through
the real OCR factory (MLX-VLM primary, no mocks), and prints the
extracted text.

Run from repo root:

    uv run python scripts/smoke_v060_ocr.py

First call downloads the MLX-VLM weights if not cached (~5 GB). On a
warm cache the run takes ~5-10 s end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from textwrap import dedent

from PIL import Image, ImageDraw, ImageFont

from sdet_brain.config import Settings
from sdet_brain.ingestion.image_parser import parse_image
from sdet_brain.ocr.factory import get_ocr_engine, reset_ocr_engine


def _render_receipt(path: Path) -> None:
    """Write a synthetic Polish receipt to ``path`` as a PNG."""
    text = dedent(
        """\
        BIEDRONKA Sklep nr 1234
        ul. Krakowska 15, 00-123 Warszawa
        NIP: 5260250274

        --------------------------------
        Mleko 2% 1L            x1   4,79
        Chleb wiejski 700g     x2   6,50
        Maslo Extra 200g       x1   8,99
        Jablka Champion 1kg    x1   3,49
        Kawa Tchibo 500g       x1  24,99
        --------------------------------

        SUMA PLN:                  48,76

        Platnosc karta:            48,76

        Dziekujemy za zakupy
        2026-05-14   16:42
        Paragon fiskalny
        """
    )

    img = Image.new("RGB", (520, 640), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Courier New.ttf", size=18,
        )
    except OSError:
        font = ImageFont.load_default()

    draw.multiline_text((20, 20), text, fill="black", font=font, spacing=6)
    img.save(path, format="PNG")


def main() -> int:
    print("== v0.6.0 OCR smoke test ==")
    print()

    # Fresh factory state — otherwise this script picks up whatever
    # cached selection a prior import path warmed.
    reset_ocr_engine()
    settings = Settings()

    print(f"Provider chain head: {settings.ocr_provider}")
    print(f"MLX model:           {settings.ocr_mlx_vlm_model}")
    print(f"Ollama primary:      {settings.ocr_ollama_primary_model}")
    print(f"Ollama fallback:     {settings.ocr_ollama_fallback_model}")
    print()

    tmp_root = Path("/tmp/sdet-brain-smoke-v060")
    tmp_root.mkdir(parents=True, exist_ok=True)
    receipt_path = tmp_root / "receipt.png"

    print(f"Rendering synthetic receipt → {receipt_path}")
    _render_receipt(receipt_path)
    print(f"Receipt size: {receipt_path.stat().st_size} bytes")
    print()

    print("Building OCR engine (factory walk + health checks)…")
    t0 = time.time()
    selection = get_ocr_engine(settings)
    print(f"  ✓ provider:  {selection.provider}")
    print(f"  ✓ model:     {selection.model}")
    print(f"  ✓ fell_back: {selection.fell_back}")
    print(f"  ✓ attempted: {selection.attempted}")
    print(f"  ✓ boot:      {time.time() - t0:.2f}s")
    print()

    print("Running OCR via image_parser.parse_image…")
    t1 = time.time()
    doc = parse_image(
        receipt_path, ocr_engine=selection.engine, settings=settings,
    )
    elapsed = time.time() - t1
    print(f"  ✓ extract took: {elapsed:.2f}s")
    print(f"  ✓ chunks:       {len(doc.chunks)}")
    print(f"  ✓ content_hash: {doc.content_hash[:12]}…")
    print(f"  ✓ frontmatter:")
    for k, v in doc.frontmatter.items():
        print(f"      {k}: {v}")
    print()

    print("=== OCR text (first chunk) ===")
    print(doc.chunks[0].text)
    print("===")

    # Sanity asserts — keywords we expect from the rendered receipt.
    expected_anywhere = ["BIEDRONKA", "SUMA", "48,76"]
    full_text = "\n".join(c.text for c in doc.chunks)
    missing = [w for w in expected_anywhere if w not in full_text]
    if missing:
        print()
        print(f"!! Missing expected tokens: {missing}")
        return 1

    print()
    print("✓ All expected tokens present. Smoke PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

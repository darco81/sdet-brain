"""Prompt-side helpers shared by every OCR provider.

DeepSeek-OCR emits grounding tokens (``<|grounding|>``, ``<|ref|>``,
``<|det|>``, ``<|image|>``) and occasionally loops on the same line.
Both quirks are post-processable with regex; the same passes work for
Qwen-VL output (no-ops when the tokens are absent).

Quality acceptance is the trigger for the factory's fallback chain:
output too short / too whitespace-y → caller raises
``OCRQualityError`` and the next engine takes over.

Ported from the production-validated Domowy Kombajn pipeline
(`m5_service/extract.py:34-62`).
"""

from __future__ import annotations

import re

DEEPSEEK_TOKEN_RE = re.compile(
    r"<\|(ref|det|grounding|image)\|>.*?<\|/(ref|det|grounding|image)\|>",
    re.DOTALL,
)
"""Matches DeepSeek-OCR grounding tokens with their content payload.

Pattern handles bounding-box markers like
``<|ref|>napis<|/ref|><|det|>[[12,34,56,78]]<|/det|>`` plus the
``<|grounding|>`` / ``<|image|>`` envelopes. ``re.DOTALL`` lets the
content span newlines, which happens with multi-line ref blocks.
"""

_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")


def strip_deepseek_tokens(text: str) -> str:
    """Strip DeepSeek-OCR grounding tokens and collapse blank-line runs.

    Safe to call on output from any VLM — no-op when the tokens are
    absent (e.g. Qwen2.5-VL produces plain markdown without grounding
    tags).
    """
    cleaned = DEEPSEEK_TOKEN_RE.sub("", text)
    return _BLANK_LINE_RUN_RE.sub("\n\n", cleaned).strip()


def deduplicate_repeats(text: str, *, max_consecutive: int = 2) -> str:
    """Collapse runs of identical consecutive lines.

    DeepSeek-OCR occasionally falls into a loop and repeats the same
    line dozens of times. After the configured ``max_consecutive``
    appearances (default: 2), subsequent identical lines are dropped.
    Lines that differ from the previous reset the counter.
    """
    if max_consecutive < 1:
        raise ValueError("max_consecutive must be >= 1")

    lines = text.split("\n")
    deduped: list[str] = []
    prev: str | None = None
    repeat_count = 0
    for line in lines:
        if line == prev:
            repeat_count += 1
            if repeat_count >= max_consecutive:
                continue
        else:
            repeat_count = 0
            prev = line
        deduped.append(line)
    return "\n".join(deduped)


def quality_acceptable(text: str, *, min_chars: int) -> bool:
    """Return ``True`` if ``text`` clears the quality bar.

    Heuristics (see plan §Quality-fallback trigger):

    * length below ``min_chars`` (after stripping leading/trailing whitespace) → False
    * text empty after strip → False
    * text contains no alphanumeric characters → False (pure punctuation/whitespace)

    Callers wrap an OCR call with these checks; a False return surfaces
    as ``OCRQualityError`` to the factory's fallback chain.
    """
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")

    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    return any(ch.isalnum() for ch in stripped)

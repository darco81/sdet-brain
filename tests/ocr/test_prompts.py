"""Pure-function coverage for the shared OCR prompt helpers."""

from __future__ import annotations

import pytest

from sdet_brain.ocr.prompts import (
    deduplicate_repeats,
    quality_acceptable,
    strip_deepseek_tokens,
)

# --- strip_deepseek_tokens --------------------------------------------------


def test_strip_removes_ref_det_tokens() -> None:
    raw = "Sklep <|ref|>napis<|/ref|><|det|>[[1,2,3,4]]<|/det|> reszta tekstu"
    assert strip_deepseek_tokens(raw) == "Sklep  reszta tekstu"


def test_strip_removes_grounding_envelope() -> None:
    raw = "before <|grounding|>some grounded chunk<|/grounding|> after"
    assert strip_deepseek_tokens(raw) == "before  after"


def test_strip_handles_image_token_multiline() -> None:
    raw = "head\n<|image|>line1\nline2\nline3<|/image|>\ntail"
    assert strip_deepseek_tokens(raw) == "head\n\ntail"


def test_strip_collapses_multiple_blank_lines() -> None:
    raw = "first\n\n\n\n\nsecond"
    assert strip_deepseek_tokens(raw) == "first\n\nsecond"


def test_strip_passthrough_plain_text() -> None:
    raw = "Just normal markdown.\n## Heading\nText."
    assert strip_deepseek_tokens(raw) == raw


# --- deduplicate_repeats ----------------------------------------------------


def test_dedup_collapses_runs_after_default_threshold() -> None:
    raw = "A\nA\nA\nA\nA\nB"
    # Default max_consecutive=2 → keep first 2 identical lines, drop the rest.
    assert deduplicate_repeats(raw) == "A\nA\nB"


def test_dedup_leaves_short_runs_intact() -> None:
    raw = "A\nA\nB\nB\nC"
    assert deduplicate_repeats(raw) == raw


def test_dedup_empty_input() -> None:
    assert deduplicate_repeats("") == ""


def test_dedup_custom_threshold_one_keeps_only_first() -> None:
    raw = "X\nX\nX\nY"
    assert deduplicate_repeats(raw, max_consecutive=1) == "X\nY"


def test_dedup_rejects_zero_threshold() -> None:
    with pytest.raises(ValueError, match="max_consecutive must be >= 1"):
        deduplicate_repeats("x\nx", max_consecutive=0)


# --- quality_acceptable -----------------------------------------------------


def test_quality_accepts_sufficient_alnum_text() -> None:
    assert quality_acceptable("Receipt total 24,99 PLN", min_chars=10)


def test_quality_rejects_too_short() -> None:
    assert quality_acceptable("hi", min_chars=10) is False


def test_quality_rejects_whitespace_only() -> None:
    assert quality_acceptable("   \n\t  ", min_chars=0) is False


def test_quality_rejects_punctuation_only() -> None:
    assert quality_acceptable("... !!! ???", min_chars=5) is False


def test_quality_accepts_at_min_char_boundary() -> None:
    assert quality_acceptable("abcde", min_chars=5)


def test_quality_rejects_negative_min_chars() -> None:
    with pytest.raises(ValueError, match="min_chars must be >= 0"):
        quality_acceptable("ok", min_chars=-1)

"""Reranker tests using a deterministic in-process fake.

The real `FastembedReranker` downloads ~80 MB - 1 GB of ONNX weights
from Hugging Face on first call. Tests must not depend on that, so
we exercise the public API through a hand-rolled fake encoder while
keeping `FastembedReranker.rerank()` logic real.
"""

from __future__ import annotations

import pytest

from sdet_brain.embeddings.reranker import (
    DEFAULT_MODEL,
    FastembedReranker,
    IReranker,
    RerankCandidate,
    RerankerError,
    RerankResult,
)


class _FakeEncoder:
    """Returns scores by string-matching tokens against the query.

    Cosine-style behaviour: a document with more shared lowercase tokens
    with the query gets a higher score. Deterministic, no I/O.
    """

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def rerank(self, query: str, documents):  # type: ignore[no-untyped-def]
        if self._fail:
            raise RuntimeError("simulated encoder failure")
        query_tokens = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_tokens = set(doc.lower().split())
            scores.append(float(len(query_tokens & doc_tokens)))
        return scores


def _build_reranker(*, fake: _FakeEncoder | None = None) -> FastembedReranker:
    rr = FastembedReranker(model_name="test-fake")
    rr._encoder = fake or _FakeEncoder()  # test injection
    return rr


def test_reranker_satisfies_protocol() -> None:
    rr = FastembedReranker(model_name="test-fake")
    assert isinstance(rr, IReranker)
    assert rr.model_name == "test-fake"


def test_rerank_reorders_by_score_and_truncates() -> None:
    rr = _build_reranker()
    candidates = [
        RerankCandidate(text="apple banana cherry", payload="A"),  # 1 match
        RerankCandidate(text="banana orange", payload="B"),  # 0 match
        RerankCandidate(text="apple", payload="C"),  # 1 match
    ]
    results = rr.rerank("apple", candidates, top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, RerankResult) for r in results)
    assert {r.payload for r in results} == {"A", "C"}
    # Scores in non-increasing order.
    assert results[0].score >= results[1].score


def test_rerank_uses_default_top_k_when_unspecified() -> None:
    rr = FastembedReranker(model_name="test-fake", top_k_default=2)
    rr._encoder = _FakeEncoder()
    candidates = [RerankCandidate(text=f"alpha doc-{i}") for i in range(5)]
    results = rr.rerank("alpha", candidates)
    assert len(results) == 2


def test_rerank_empty_candidates_returns_empty() -> None:
    rr = _build_reranker()
    assert rr.rerank("anything", []) == []


def test_rerank_empty_query_raises() -> None:
    rr = _build_reranker()
    with pytest.raises(RerankerError):
        rr.rerank("   ", [RerankCandidate(text="x")])


def test_rerank_encoder_failure_wraps_error() -> None:
    rr = _build_reranker(fake=_FakeEncoder(fail=True))
    with pytest.raises(RerankerError):
        rr.rerank("q", [RerankCandidate(text="x")])


def test_rerank_score_count_mismatch_raises() -> None:
    class _MismatchEncoder:
        def rerank(self, query, documents):  # type: ignore[no-untyped-def]
            return [0.0]  # always 1 score regardless of input

    rr = _build_reranker(fake=_MismatchEncoder())  # type: ignore[arg-type]
    with pytest.raises(RerankerError):
        rr.rerank(
            "q",
            [RerankCandidate(text="a"), RerankCandidate(text="b")],
        )


def test_rerank_preserves_payload_through_reorder() -> None:
    rr = _build_reranker()
    candidates = [
        RerankCandidate(text="alpha beta", payload={"id": 1}),
        RerankCandidate(text="alpha gamma", payload={"id": 2}),
    ]
    results = rr.rerank("alpha", candidates)
    payload_ids = sorted(r.payload["id"] for r in results)
    assert payload_ids == [1, 2]


def test_health_check_returns_true_on_success() -> None:
    rr = _build_reranker()
    assert rr.health_check() is True


def test_health_check_returns_false_on_failure() -> None:
    rr = _build_reranker(fake=_FakeEncoder(fail=True))
    assert rr.health_check() is False


def test_default_model_is_multilingual() -> None:
    """Sanity: the default reranker must be PL+EN aware."""
    assert "multilingual" in DEFAULT_MODEL.lower()

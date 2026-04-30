"""Protocol contract checks - no provider runtime needed."""

from __future__ import annotations

from sdet_brain.embeddings.protocol import IEmbedder


class _FakeEmbedder:
    vector_size = 4
    model_name = "fake/embed"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.1, 0.2, 0.3] for _ in texts]

    def health_check(self) -> bool:
        return True


def test_runtime_isinstance_check_accepts_compatible_class() -> None:
    assert isinstance(_FakeEmbedder(), IEmbedder)


def test_fake_embedder_round_trip_shape() -> None:
    embedder = _FakeEmbedder()
    output = embedder.embed(["a", "b"])
    assert len(output) == 2
    assert all(len(vec) == embedder.vector_size for vec in output)

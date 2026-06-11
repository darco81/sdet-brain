"""SDET Brain - persistent RAG for SDET brand domain."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sdet-brain")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.0.0.dev0"

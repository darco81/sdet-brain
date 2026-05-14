"""OllamaOCREngine tests — exercised with monkeypatched httpx primitives."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from sdet_brain.ocr.ollama_provider import OllamaOCREngine
from sdet_brain.ocr.protocol import (
    OCRError,
    OCRQualityError,
    OCRResult,
    OCRTimeoutError,
)


def _ok_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=payload,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
    )


@pytest.fixture
def engine() -> OllamaOCREngine:
    return OllamaOCREngine(
        model_name="deepseek-ocr",
        default_prompt="Convert the document.",
        quality_min_chars=10,
        keep_alive="5m",
        timeout_seconds=120,
    )


def test_extract_text_returns_ocr_result(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_a, **_kw: _ok_response({"response": "Receipt total 24,99 PLN"}),
    )

    result = engine.extract_text(b"\x89PNG fake")

    assert isinstance(result, OCRResult)
    assert result.text == "Receipt total 24,99 PLN"
    assert result.model == "ollama:deepseek-ocr"
    assert result.peak_memory_gb is None
    assert result.duration_s >= 0


def test_extract_text_base64_encodes_payload(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        url: str, *, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        captured["url"] = url
        captured["payload"] = json
        captured["timeout"] = timeout
        return _ok_response({"response": "long enough output text content"})

    monkeypatch.setattr(httpx, "post", fake_post)

    engine.extract_text(b"\x89PNG fake")

    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 120.0
    payload = captured["payload"]
    assert payload["model"] == "deepseek-ocr"
    assert payload["stream"] is False
    assert payload["keep_alive"] == "5m"
    assert payload["options"]["temperature"] == 0.0
    assert payload["images"] == [base64.b64encode(b"\x89PNG fake").decode("ascii")]


def test_extract_text_strips_and_dedupes(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    raw = (
        "Header content full line A\n"
        "Header content full line A\n"
        "Header content full line A\n"
        "Header content full line A\n"
        "<|ref|>noisy<|/ref|><|det|>[[1,2]]<|/det|> Real data here"
    )
    monkeypatch.setattr(
        httpx, "post", lambda *_a, **_kw: _ok_response({"response": raw}),
    )

    result = engine.extract_text(b"img")
    assert "<|ref|>" not in result.text
    assert "<|/det|>" not in result.text
    assert result.text.count("Header content full line A") == 2
    assert "Real data here" in result.text


def test_extract_text_raises_quality_error_when_too_short(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    monkeypatch.setattr(
        httpx, "post", lambda *_a, **_kw: _ok_response({"response": "hi"}),
    )

    with pytest.raises(OCRQualityError) as excinfo:
        engine.extract_text(b"img")
    assert "deepseek-ocr" in str(excinfo.value)
    assert "min=10" in str(excinfo.value)


def test_extract_text_raises_timeout_on_httpx_timeout(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    def fake_post(*_a: Any, **_kw: Any) -> httpx.Response:
        raise httpx.ReadTimeout("simulated read timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OCRTimeoutError) as excinfo:
        engine.extract_text(b"img")
    assert "120s" in str(excinfo.value)
    assert "deepseek-ocr" in str(excinfo.value)


def test_extract_text_raises_ocr_error_on_connection_failure(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    def fake_post(*_a: Any, **_kw: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OCRError) as excinfo:
        engine.extract_text(b"img")
    assert "deepseek-ocr" in str(excinfo.value)


def test_extract_text_rejects_empty_bytes(engine: OllamaOCREngine) -> None:
    with pytest.raises(OCRError, match="Empty image_bytes"):
        engine.extract_text(b"")


def test_extract_text_passes_custom_prompt(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(
        _url: str, *, json: dict[str, Any], timeout: float
    ) -> httpx.Response:
        _ = timeout
        captured["prompt"] = json["prompt"]
        return _ok_response({"response": "long enough output content here"})

    monkeypatch.setattr(httpx, "post", fake_post)

    engine.extract_text(b"img", prompt="Custom prompt please.")
    assert captured["prompt"] == "Custom prompt please."


def test_health_check_returns_true_on_200(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, timeout=None: httpx.Response(
            status_code=200,
            json={"models": []},
            request=httpx.Request("GET", url),
        ),
    )
    assert engine.health_check() is True


def test_health_check_returns_false_on_connect_error(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    def fake_get(*_a: Any, **_kw: Any) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", fake_get)
    assert engine.health_check() is False


def test_health_check_returns_false_on_5xx(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, timeout=None: httpx.Response(
            status_code=502,
            request=httpx.Request("GET", url),
        ),
    )
    assert engine.health_check() is False


def test_custom_host_strips_trailing_slash() -> None:
    engine = OllamaOCREngine(
        model_name="qwen2.5-vl:32b",
        default_prompt="Extract text.",
        quality_min_chars=10,
        host="http://example.com:11434/",
    )
    assert engine.model_name == "ollama:qwen2.5-vl:32b"


def test_trailing_slash_host_produces_single_slash_request_url(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: trailing-slash host must not leak into the request URL."""
    captured: dict[str, str] = {}

    def fake_post(url: str, *, json: Any, timeout: float) -> httpx.Response:
        _ = json, timeout
        captured["url"] = url
        return _ok_response({"response": "long enough output text here"})

    monkeypatch.setattr(httpx, "post", fake_post)
    engine = OllamaOCREngine(
        model_name="deepseek-ocr",
        default_prompt="Convert.",
        quality_min_chars=10,
        host="http://example.com:11434/",
    )
    engine.extract_text(b"img")
    assert captured["url"] == "http://example.com:11434/api/generate"


@pytest.mark.parametrize("status_code", [404, 422, 500, 502])
def test_extract_text_4xx_5xx_response_raises_ocr_error(
    monkeypatch: pytest.MonkeyPatch,
    engine: OllamaOCREngine,
    status_code: int,
) -> None:
    """Production-realistic HTTP failures (model-not-found 404, bad
    gateway 502, etc.) should raise OCRError with the model name in
    the message."""
    request = httpx.Request("POST", "http://localhost:11434/api/generate")

    def fake_post(*_a: Any, **_kw: Any) -> httpx.Response:
        return httpx.Response(status_code=status_code, request=request)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OCRError) as excinfo:
        engine.extract_text(b"img")
    assert "deepseek-ocr" in str(excinfo.value)


def test_extract_text_non_json_response_raises_ocr_error(
    monkeypatch: pytest.MonkeyPatch, engine: OllamaOCREngine
) -> None:
    """When Ollama sits behind a reverse proxy that returns HTML error
    pages, the JSON decode fails — surface as OCRError, not raw ValueError."""

    def fake_post(*_a: Any, **_kw: Any) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"<html><body>504 gateway</body></html>",
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(OCRError) as excinfo:
        engine.extract_text(b"img")
    assert "non-JSON" in str(excinfo.value)


def test_health_check_logs_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
    engine: OllamaOCREngine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Diagnostic improvement: log the exception itself, not just the URL,
    so the operator can tell ConnectError from ProxyError from 5xx."""
    import logging

    def fake_get(*_a: Any, **_kw: Any) -> httpx.Response:
        raise httpx.ConnectError("Connection refused — daemon down?")

    monkeypatch.setattr(httpx, "get", fake_get)
    with caplog.at_level(logging.WARNING):
        result = engine.health_check()

    assert result is False
    # The exception message must appear in the log so triage is possible.
    assert any("daemon down" in rec.message for rec in caplog.records)

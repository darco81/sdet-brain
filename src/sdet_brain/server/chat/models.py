"""Pydantic request/response models for the chat endpoint."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatTurn(BaseModel):
    """One message in the conversation history."""

    role: Role
    content: Annotated[str, Field(min_length=1)]


class ChatRequest(BaseModel):
    """Inbound payload for ``POST /chat``.

    The server is stateless: each request carries the full conversation
    history. ``stream=true`` switches the response to Server-Sent Events
    (one ``data:`` frame per token).
    """

    messages: Annotated[list[ChatTurn], Field(min_length=1)]
    stream: bool = False
    retrieve: bool = Field(
        default=True,
        description=(
            "When True (default), hybrid-search the latest user turn and "
            "inject the top chunks as retrieved context for the LLM."
        ),
    )
    top_k: Annotated[int, Field(ge=1, le=20)] = 6
    max_tokens: Annotated[int, Field(ge=16, le=2048)] = 512


class ChatResponse(BaseModel):
    """Non-streaming response shape."""

    reply: str
    sources: list[str] = Field(default_factory=list)
    retrieved_chunk_count: int = 0

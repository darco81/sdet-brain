"""`sdet-brain-chat` - terminal REPL backed by the live /chat endpoint (T5-01).

prompt_toolkit drives the multi-line input + history file. httpx
streams the SSE response and we parse `data:` frames manually so
each token can render incrementally. Slash commands give the user
control over conversation state (`/clear`, `/save`, `/load`,
`/sources`, `/help`, `/quit`).

The brain server is expected to be running on
``http://localhost:8080`` by default. Override with ``BRAIN_URL`` env
var or ``--url``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("sdet_brain.cli.chat_repl")

DEFAULT_URL = "http://localhost:8080"
HOME_DIR = Path.home() / ".sdet-brain"
HISTORY_FILE = HOME_DIR / "chat_history"
CONVERSATIONS_DIR = HOME_DIR / "conversations"


@dataclass
class ReplState:
    """In-memory state for one REPL session.

    Decoupled from prompt_toolkit / httpx so unit tests can drive the
    parser + dispatcher without touching the terminal or the network.
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    last_sources: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.messages.clear()
        self.last_sources.clear()


@dataclass(frozen=True)
class CommandResult:
    """What a slash-command handler tells the REPL to do.

    ``output`` is printed to the user. ``should_exit`` flips the REPL
    loop off. ``send_to_llm`` is the user message body when the input
    wasn't a slash command (`None` for handled commands).
    """

    output: str = ""
    should_exit: bool = False
    send_to_llm: str | None = None


HELP_TEXT = """\
Slash commands:
  /help              this help
  /clear             reset the conversation history
  /sources           print structured sources from the last response
  /save NAME         dump current conversation to ~/.sdet-brain/conversations/NAME.json
  /load NAME         restore a saved conversation
  /quit, /exit       leave the REPL
Anything else is sent to the brain as the next user turn.
"""


def parse_command(line: str, state: ReplState) -> CommandResult:
    """Parse a single REPL line and decide what to do.

    Pure function (no I/O beyond reading the conversation file in
    `/load`) so tests can drive every branch without spinning up
    prompt_toolkit or httpx.
    """
    raw = line.strip()
    if not raw:
        return CommandResult()
    if not raw.startswith("/"):
        return CommandResult(send_to_llm=raw)

    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/?"):
        return CommandResult(output=HELP_TEXT)
    if cmd == "/clear":
        state.clear()
        return CommandResult(output="(conversation cleared)")
    if cmd == "/sources":
        if not state.last_sources:
            return CommandResult(output="(no sources from the last response)")
        return CommandResult(output=_format_sources(state.last_sources))
    if cmd == "/save":
        if not arg:
            return CommandResult(output="usage: /save NAME")
        path = _save_conversation(arg, state)
        return CommandResult(output=f"(saved to {path})")
    if cmd == "/load":
        if not arg:
            return CommandResult(output="usage: /load NAME")
        try:
            _load_conversation(arg, state)
        except FileNotFoundError:
            return CommandResult(output=f"(no saved conversation named {arg!r})")
        return CommandResult(output=f"(loaded {len(state.messages)} messages from {arg!r})")
    if cmd in ("/quit", "/exit"):
        return CommandResult(output="bye.", should_exit=True)
    return CommandResult(output=f"(unknown command {cmd!r}; try /help)")


def _format_sources(sources: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for src in sources:
        n = src.get("n", "?")
        path = src.get("source_path", "(unknown)")
        score = src.get("score")
        score_str = f" score={score:.3f}" if isinstance(score, int | float) else ""
        snippet = src.get("snippet", "")
        head = f"[{n}] {path}{score_str}"
        lines.append(head)
        if snippet:
            lines.append(f"    {snippet}")
    return "\n".join(lines)


def _save_conversation(name: str, state: ReplState) -> Path:
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = CONVERSATIONS_DIR / f"{name}.json"
    payload = {
        "messages": list(state.messages),
        "last_sources": list(state.last_sources),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _load_conversation(name: str, state: ReplState) -> None:
    path = CONVERSATIONS_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    state.messages = list(raw.get("messages", []))
    state.last_sources = list(raw.get("last_sources", []))


def stream_chat(
    url: str,
    messages: list[dict[str, str]],
    *,
    timeout: float = 600.0,
) -> Iterator[dict[str, Any]]:
    """Yield SSE-decoded payloads from ``POST /chat`` with ``stream=true``."""
    payload: dict[str, Any] = {"messages": messages, "stream": True}
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream("POST", f"{url}/chat", json=payload) as response,
    ):
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            try:
                yield json.loads(line[len("data: ") :])
            except json.JSONDecodeError as exc:
                logger.warning("malformed SSE frame: %s", exc)


def _run_turn(url: str, state: ReplState, body: str) -> None:
    state.messages.append({"role": "user", "content": body})
    assembled: list[str] = []
    sources: list[dict[str, Any]] = []
    print()  # leading newline before streaming output
    try:
        for frame in stream_chat(url, state.messages):
            if frame.get("event") == "done":
                sources = list(frame.get("sources", []))
                break
            text = frame.get("text", "")
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()
                assembled.append(text)
    except httpx.HTTPError as exc:
        print(f"\n(error talking to brain: {exc})", file=sys.stderr)
        # roll back the user message so a retry doesn't double-up
        state.messages.pop()
        return
    print()  # trailing newline
    if assembled:
        state.messages.append({"role": "assistant", "content": "".join(assembled)})
    state.last_sources = sources


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain-chat",
        description="Terminal REPL for the SDET Brain /chat endpoint.",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("BRAIN_URL", DEFAULT_URL),
        help="Brain server base URL (default $BRAIN_URL or http://localhost:8080).",
    )
    return parser


def _make_session() -> Any:
    """Build the prompt_toolkit session with persistent history."""
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    return PromptSession(history=FileHistory(str(HISTORY_FILE)))


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = _build_parser().parse_args(argv)
    url = args.url

    state = ReplState()
    print(
        f"sdet-brain-chat → {url}\nType /help for commands, /quit to exit.\n",
        file=sys.stderr,
    )
    try:
        session = _make_session()
    except ImportError as exc:  # pragma: no cover - dep guard
        print(f"prompt_toolkit unavailable: {exc}", file=sys.stderr)
        return 2

    while True:
        try:
            line = session.prompt("> ")
        except (EOFError, KeyboardInterrupt):
            print("bye.", file=sys.stderr)
            return 0
        result = parse_command(line, state)
        if result.output:
            print(result.output)
        if result.should_exit:
            return 0
        if result.send_to_llm is not None:
            _run_turn(url, state, result.send_to_llm)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

#!/usr/bin/env -S uv run --script python3
"""Daily LLM digest - 'what's new in drafts in the last 24h'.

Pulls drafts chunks from Qdrant where ``created_at >= now - 24h``,
feeds them to the local MLX Qwen3-Next-80B Instruct, writes the
result to ``~/Documents/sdet-digests/YYYY-MM-DD.md`` (append-style
changelog) and ships an embed to Discord. The ``uv run --script``
shebang means the daemon can call the file directly without
remembering the venv.

Cold-start by design: this script imports the LLM router fresh, runs
once, exits. The 80B weights leave RAM with the process. Daemon
keeps only its small embedder warm.

Required env: ``DISCORD_WEBHOOK_URL`` (loaded by daily.sh from
``~/.config/sdet-brain/discord.env`` - never committed).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.request

from qdrant_client.models import FieldCondition, Filter, MatchValue

from sdet_brain.config import get_settings
from sdet_brain.llm.factory import get_router
from sdet_brain.llm.protocol import ChatMessage
from sdet_brain.storage.qdrant_client import QdrantStorage

DIGEST_DIR = pathlib.Path(
    os.environ.get(
        "SDET_DIGEST_DIR",
        str(pathlib.Path.home() / "Documents" / "sdet-digests"),
    )
)
LOG = pathlib.Path(os.environ.get("SDET_DIGEST_LOG", "/tmp/sdet-brain-digest.log"))
WINDOW_HOURS = int(os.environ.get("SDET_DIGEST_WINDOW_HOURS", "24"))
MAX_PASSAGES = int(os.environ.get("SDET_DIGEST_MAX_PASSAGES", "12"))
SOURCE_TYPE = os.environ.get("SDET_DIGEST_SOURCE_TYPE", "drafts")
DIGEST_LANG = os.environ.get("SDET_DIGEST_LANG", "pl")

_PROMPT_PL = (
    "Jesteś redaktorem digestu. Dostajesz numerowane fragmenty z "
    f"materiałów (source_type={SOURCE_TYPE!r}) dodanych w ostatnich "
    f"{WINDOW_HOURS}h. Napisz zwięzły, konkretny digest po polsku: "
    "3-5 punktów, każdy 1-2 zdania, cytuj źródła jako [n]. Bez "
    "marketingowego pierdolenia. Jeśli fragmenty są błahe albo "
    "techniczne, powiedz to wprost."
)
_PROMPT_EN = (
    "You are a digest editor. You receive numbered passages from "
    f"materials (source_type={SOURCE_TYPE!r}) added in the last "
    f"{WINDOW_HOURS}h. Write a concise, concrete digest in English: "
    "3-5 bullets, each 1-2 sentences, cite sources as [n]. No "
    "marketing fluff. If the passages are trivial or just technical "
    "notes, say so plainly."
)
SYSTEM_PROMPT = _PROMPT_PL if DIGEST_LANG.startswith("pl") else _PROMPT_EN


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {message}\n")


def banner(title: str, subtitle: str, message: str, sound: str | None = None) -> None:
    sound_clause = f" sound name \"{sound}\"" if sound else ""
    script = (
        f'display notification "{message}" '
        f'with title "{title}" subtitle "{subtitle}"{sound_clause}'
    )
    subprocess.run(["/usr/bin/osascript", "-e", script], check=False)


def fetch_recent_draft_chunks(storage: QdrantStorage, settings) -> list[dict]:
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=WINDOW_HOURS)
    cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    flt = Filter(
        must=[FieldCondition(key="source_type", match=MatchValue(value=SOURCE_TYPE))]
    )
    points: list[dict] = []
    next_offset = None
    while True:
        batch, next_offset = storage._client.scroll(
            collection_name=settings.collection_name,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        for p in batch:
            payload = dict(p.payload or {})
            created = payload.get("created_at", "")
            if created and created >= cutoff_iso:
                points.append(payload)
        if next_offset is None:
            break
    points.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return points[:MAX_PASSAGES]


def build_prompt(passages: list[dict]) -> str:
    lines = []
    for idx, p in enumerate(passages, start=1):
        path = p.get("source_path", "(unknown)")
        heading = p.get("heading_path", "")
        text = (p.get("text") or "").strip()
        head = f"[{idx}] [{path}]"
        if heading:
            head += f" - {heading}"
        lines.append(f"{head}\n{text}")
    return "Fragmenty:\n\n" + "\n\n".join(lines)


def write_changelog(today: str, summary: str, passages: list[dict]) -> pathlib.Path:
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    path = DIGEST_DIR / f"{today}.md"
    sources = sorted({p.get("source_path", "") for p in passages if p.get("source_path")})
    body = (
        f"# Digest {today}\n\n"
        f"**Window:** ostatnie {WINDOW_HOURS}h · **Fragmentów:** {len(passages)}\n\n"
        f"{summary.strip()}\n\n"
        f"## Źródła\n\n"
        + "\n".join(f"- `{s}`" for s in sources)
        + "\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def post_discord(webhook_url: str, today: str, summary: str, passages: list[dict]) -> bool:
    if not webhook_url:
        log("no DISCORD_WEBHOOK_URL - skipping discord")
        return False
    sources = sorted({p.get("source_path", "") for p in passages if p.get("source_path")})
    short_sources = [pathlib.Path(s).name for s in sources][:8]
    embed = {
        "title": f"sdet-brain digest · {today}",
        "description": summary[:4000],
        "color": 0x6E5BFE,
        "fields": [
            {
                "name": f"Fragmentów: {len(passages)}",
                "value": "\n".join(f"`{s}`" for s in short_sources) or "-",
                "inline": False,
            }
        ],
        "footer": {"text": f"window: {WINDOW_HOURS}h · MLX Qwen3-Next-80B"},
    }
    payload = {"username": "sdet-brain", "embeds": [embed]}
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            # Cloudflare returns 1010 (browser fingerprint blocked) without a
            # real User-Agent, so masquerade as a normal client.
            "User-Agent": "sdet-brain-digest/1.0 (+https://sdet.it)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            log(f"discord posted status={resp.status}")
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as exc:
        log(f"discord HTTP error: {exc.code} {exc.read()[:200]!r}")
    except urllib.error.URLError as exc:
        log(f"discord URL error: {exc}")
    return False


def main() -> int:
    today = dt.date.today().isoformat()
    log(f"=== digest start {today} ===")
    settings = get_settings()
    webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()

    with QdrantStorage(settings.qdrant_url, api_key=settings.qdrant_api_key) as storage:
        passages = fetch_recent_draft_chunks(storage, settings)

    log(f"matched {len(passages)} chunks in last {WINDOW_HOURS}h")
    if not passages:
        msg = (
            f"Brak nowych fragmentów ({SOURCE_TYPE}) w ostatnich {WINDOW_HOURS}h."
            if DIGEST_LANG.startswith("pl")
            else f"No new {SOURCE_TYPE} passages in the last {WINDOW_HOURS}h."
        )
        path = write_changelog(today, msg, [])
        banner("sdet-brain digest", today, msg)
        post_discord(webhook, today, msg, [])
        log(f"empty digest written to {path}")
        return 0

    prompt = build_prompt(passages)
    router = get_router()
    log(f"calling LLM (cold start) - model tier=summarize")
    try:
        summary = router.chat(
            [
                ChatMessage(role="system", content=SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            task="summarize",
            max_tokens=600,
            temperature=0.4,
        ).strip()
    except Exception as exc:  # noqa: BLE001 - log and surface in banner
        log(f"LLM failed: {exc}")
        banner("sdet-brain digest FAIL", today, str(exc)[:120], sound="Basso")
        return 1

    if not summary:
        summary = "_(LLM zwrócił pustkę)_"
    path = write_changelog(today, summary, passages)
    log(f"wrote digest to {path}")

    short = summary.splitlines()[0][:120] if summary else "(pusty digest)"
    banner("sdet-brain digest", today, f"{len(passages)} fragmentów · {short}")
    post_discord(webhook, today, summary, passages)
    log(f"=== digest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

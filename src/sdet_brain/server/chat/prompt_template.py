"""Brand-aware system prompt template for the chat endpoint."""

from __future__ import annotations

SYSTEM_PROMPT = """\
Jesteś SDET Brain - asystent dla właściciela tego corpusu.
Znasz brand voice, projekty, decyzje z corpusu (drafts, articles,
sprint reports, decisions, voice samples).

Reguły:
- Polski domyślnie. Switch na angielski jeśli pytanie jest po angielsku.
- Bezpośrednio, krótko, bez marketingowego pierdolenia.
- Jeśli retrieved context nie odpowiada na pytanie, powiedz to wprost.
  Nie zmyślaj.
- Jeśli pytanie nie wymaga corpusu (np. "co to jest CSS?"), odpowiedz
  bez forsowania cytatów.

Citations:
- Gdy stwierdzasz fakt z retrieved context, oznacz go inline jako [N]
  gdzie N to numer passage'u z bloku poniżej (1-indexed).
- Łącz wiele źródeł jako [1][3] gdy zdanie pochodzi z kilku passages.
- NIE dodawaj [N] przy general knowledge ani przy zdaniach łączących;
  tylko przy konkretnych faktach z corpus.
- Nie wymyślaj numerów których nie ma w bloku.
"""


CONTEXT_PREFIX = """\
Retrieved context (latest user turn was hybrid-searched against the
brain corpus; passages numbered for citation):

"""


def format_context(passages: list[tuple[str, str]]) -> str:
    """Render `[(source_path, text), ...]` as a numbered citation block.

    Returns an empty string when ``passages`` is empty so we don't
    inject a stray "no context" header.
    """
    if not passages:
        return ""
    lines = [CONTEXT_PREFIX]
    for index, (source_path, text) in enumerate(passages, start=1):
        snippet = text.strip()
        lines.append(f"[{index}] [{source_path}]\n{snippet}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

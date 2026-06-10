"""`sdet-brain run` and `sdet-brain template ...` CLI entry points (T5-03).

Subcommands:

* ``run NAME [--var KEY=VAL ...]`` - render the named template's
  ``query_template`` with the given variables and dispatch the
  template's tool against the local app state.
* ``template list`` - list discoverable templates.
* ``template show NAME`` - print the YAML for ``NAME``.

Tools dispatch through the in-process `AppState` so the CLI doesn't
need the FastAPI server running. The first call still pays the MLX /
fastembed cold starts; subsequent calls reuse the warm caches.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sdet_brain.cli.templates import (
    QueryTemplate,
    TemplateError,
    discover_templates,
    find_template,
    format_output,
    render_query,
)
from sdet_brain.config import get_settings
from sdet_brain.server.dependencies import AppState
from sdet_brain.server.state import build_default_state
from sdet_brain.server.tools.domain import (
    list_articles_by_status,
    search_decisions,
    search_smaczki,
    search_sprint_reports,
    search_voice_samples,
)
from sdet_brain.server.tools.multi_query import multi_query_search
from sdet_brain.server.tools.query_rewrite import query_rewrite
from sdet_brain.server.tools.search import search
from sdet_brain.server.tools.summarize_results import summarize_results

logger = logging.getLogger("sdet_brain.cli.templates")


# Each tool gets a small adapter that pulls the rendered query / args
# out of the QueryTemplate and calls the underlying function. Keeping
# these tiny means the dispatcher stays explicit (no eval, no getattr).
_ToolFn = Callable[[AppState, str, QueryTemplate, dict[str, Any]], str]


def _run_search(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return search(
        state,
        query=q,
        limit=int(args.get("limit", 5)),
        source_type=args.get("source_type"),
        min_score=float(args.get("min_score", 0.0)),
    )


def _run_voice(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return search_voice_samples(state, topic=q, limit=int(args.get("limit", 5)))


def _run_smaczki(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return search_smaczki(state, topic=q, limit=int(args.get("limit", 5)))


def _run_decisions(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return search_decisions(
        state,
        topic=q,
        since=args.get("since"),
        limit=int(args.get("limit", 5)),
    )


def _run_sprint_reports(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return search_sprint_reports(
        state,
        query=q,
        project=args.get("project"),
        limit=int(args.get("limit", 5)),
    )


def _run_articles(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    # `list_articles_by_status` doesn't take a query; it scrolls. The
    # rendered ``q`` is ignored to keep the tool surface honest.
    _ = q
    return list_articles_by_status(
        state,
        status=str(args.get("status", "draft")),
        series=args.get("series"),
    )


def _run_query_rewrite(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return query_rewrite(state, query=q, limit=int(args.get("limit", 5)))


def _run_summarize(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return summarize_results(state, topic=q, limit=int(args.get("limit", 8)))


def _run_multi_query(state: AppState, q: str, tpl: QueryTemplate, args: dict[str, Any]) -> str:
    return multi_query_search(
        state,
        query=q,
        limit=int(args.get("limit", 5)),
        per_query_limit=int(args.get("per_query_limit", 8)),
    )


_TOOL_DISPATCH: dict[str, _ToolFn] = {
    "search": _run_search,
    "search_voice_samples": _run_voice,
    "search_smaczki": _run_smaczki,
    "search_decisions": _run_decisions,
    "search_sprint_reports": _run_sprint_reports,
    "list_articles_by_status": _run_articles,
    "query_rewrite": _run_query_rewrite,
    "summarize_results": _run_summarize,
    "multi_query_search": _run_multi_query,
}


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    """Split ``--var KEY=VAL`` arguments into a dict.

    Values are kept as strings; the underlying tool adapter coerces
    them as needed (`limit` → int, `min_score` → float, etc.).
    """
    out: dict[str, str] = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(f"--var expects KEY=VAL, got {raw!r}")
        key, value = raw.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdet-brain", description="SDET Brain CLI (templates + run)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a saved template by name.")
    run.add_argument("name")
    run.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a template variable (repeatable).",
    )
    run.add_argument(
        "--format",
        choices=("markdown", "json", "text"),
        default=None,
        help="Override the template's output_format.",
    )

    tpl = sub.add_parser("template", help="Inspect saved templates.")
    tpl_sub = tpl.add_subparsers(dest="tpl_cmd", required=True)
    tpl_sub.add_parser("list", help="List discoverable templates.")
    show = tpl_sub.add_parser("show", help="Print template YAML.")
    show.add_argument("name")

    return parser


def cmd_template_list(_: argparse.Namespace) -> int:
    locs = discover_templates()
    if not locs:
        print("No templates found.", file=sys.stderr)
        return 1
    width = max(len(loc.template.name) for loc in locs)
    for loc in locs:
        print(f"{loc.template.name:<{width}}  {loc.template.description}")
        print(f"{' ' * (width + 2)}  ({loc.path})")
    return 0


def cmd_template_show(args: argparse.Namespace) -> int:
    loc = find_template(args.name)
    print(loc.path.read_text(encoding="utf-8"))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    overrides = _parse_kv(args.var)
    loc = find_template(args.name)
    tpl = loc.template
    rendered = render_query(tpl, **overrides)

    state = build_default_state(get_settings())
    dispatcher = _TOOL_DISPATCH.get(tpl.tool)
    if dispatcher is None:
        raise TemplateError(f"unknown tool {tpl.tool!r}")

    body = dispatcher(state, rendered, tpl, dict(tpl.args))
    fmt = args.format or tpl.output_format
    print(format_output(fmt, body))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "run":
            return cmd_run(args)
        if args.cmd == "template":
            if args.tpl_cmd == "list":
                return cmd_template_list(args)
            if args.tpl_cmd == "show":
                return cmd_template_show(args)
        # argparse's required=True on subparsers makes this unreachable
        # at runtime, but keep an explicit non-zero exit just in case.
        parser.error("unreachable")
    except TemplateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["_TOOL_DISPATCH", "_parse_kv", "main"]


# Keep mypy happy about the unused Path import - tests import this
# module directly and may rely on the symbol later.
_ = Path

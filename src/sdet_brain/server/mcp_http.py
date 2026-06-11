"""MCP Streamable HTTP entrypoint - the current transport for remote clients.

Streamable HTTP (single ``/mcp`` endpoint) is the MCP transport that
replaced the deprecated HTTP+SSE transport in the 2025-03-26 spec
revision. Prefer this over ``sdet-brain-mcp-sse`` for any remote
deployment; ``sdet-brain-mcp-stdio`` remains the standard for local
Claude Desktop / Code use.
"""

from __future__ import annotations

import logging
import sys

from sdet_brain.config import get_settings
from sdet_brain.server.mcp_server import build_mcp
from sdet_brain.server.state import build_default_state


def main() -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    state = build_default_state(settings)
    mcp = build_mcp(state_getter=lambda: state)
    mcp.run(
        transport="http",
        host=settings.server_host,
        port=settings.mcp_sse_port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

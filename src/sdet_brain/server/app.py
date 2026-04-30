"""FastAPI application factory.

`create_app()` builds the app and wires its lifespan: Qdrant + embedder
construction at startup, graceful close on shutdown. Failures during
startup are logged and reported through `/health` rather than crashing
the process - the server stays useful enough to surface its own
problems.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from sdet_brain.config import Settings, get_settings
from sdet_brain.server.mcp_server import build_mcp
from sdet_brain.server.routes.health import router as health_router
from sdet_brain.server.routes.ingest import router as ingest_router
from sdet_brain.server.routes.search import router as search_router
from sdet_brain.server.routes.status import router as status_router
from sdet_brain.server.state import build_default_state

logger = logging.getLogger("sdet_brain.server")


# Re-exported under the original name so existing tests that import
# `_build_state` from this module keep working.
_build_state = build_default_state


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    state = build_default_state(settings)
    app.state.app_state = state
    logger.info(
        "Server ready (qdrant_ok=%s, embedder_ok=%s)",
        state.storage is not None,
        state.embedder is not None,
    )
    try:
        yield
    finally:
        if state.storage is not None:
            state.storage.close()
        logger.info("Server shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with the SDET Brain routes and FastMCP mount."""
    settings = settings or get_settings()
    app = FastAPI(
        title="SDET Brain",
        version="0.1.0",
        description="Persistent RAG for the SDET brand domain.",
        lifespan=_lifespan,
    )
    # The FastMCP instance is mounted before the lifespan runs, so it
    # cannot capture the AppState directly. Tools resolve state at call
    # time through this getter.
    mcp = build_mcp(state_getter=lambda: getattr(app.state, "app_state", None))

    app.include_router(health_router)
    app.include_router(status_router)
    app.include_router(search_router)
    app.include_router(ingest_router)

    # Mount the FastMCP streamable HTTP transport so any client that
    # speaks MCP-over-HTTP can reach the same tools as the stdio/SSE
    # entrypoints.
    app.mount("/mcp", mcp.http_app(transport="streamable-http"))

    # Stash the FastMCP instance on the app so tests / future tooling
    # can introspect registered tools without re-building it.
    app.state.mcp = mcp

    _ = settings  # kept for symmetry with future per-instance config wiring
    return app


def main() -> int:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    uvicorn.run(
        "sdet_brain.server.app:create_app",
        host=settings.server_host,
        port=settings.server_port,
        factory=True,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Expose QueryMind analysis as a Streamable HTTP MCP Server."""
from __future__ import annotations

import time

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.service import AnalysisService

settings = get_settings()
mcp = FastMCP("QueryMind", host=settings.mcp_host, port=settings.querymind_mcp_port)
service = AnalysisService()


@mcp.tool()
def analyze_business_data(question: str, token_budget: int = 20000) -> dict:
    """Submit a governed Northwind analysis through Redis and wait for Worker output."""
    thread = service.repo.create_thread(title=question[:28])
    run_id = service.create_analysis(thread_id=thread.id, question=question, token_budget=token_budget)
    for _ in range(600):
        run = service.repo.get_run(run_id)
        if run.status in {"completed", "completed_with_warning", "failed", "canceled"}:
            return {"run_id": run.id, "status": run.status, "answer": run.answer, "error": run.error}
        time.sleep(0.2)
    return {"run_id": run_id, "status": "timeout", "error": "等待分析结果超时"}


@mcp.tool()
def get_business_metrics() -> list[dict]:
    """Return QueryMind's governed business metric definitions."""
    from app.northwind import METRIC_DEFINITIONS
    return METRIC_DEFINITIONS


@mcp.tool()
def inspect_northwind_source() -> dict:
    """Return Northwind PostgreSQL tables, views and row counts."""
    return {"schema": service.source.schema(), "row_counts": service.source.row_counts()}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

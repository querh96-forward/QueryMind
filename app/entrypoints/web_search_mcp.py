"""External Web Search MCP Server called by QueryMind's MCP Client."""
from __future__ import annotations

import httpx
from mcp.server.fastmcp import FastMCP

from app.config import get_settings

settings = get_settings()
mcp = FastMCP("QueryMind Web Search", host=settings.mcp_host, port=settings.web_mcp_port)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> dict:
    """Search current public information through Tavily."""
    key = settings.tavily_api_key.strip()
    if not key:
        raise RuntimeError("TAVILY_API_KEY未配置")
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": key,
            "query": query,
            "max_results": max(1, min(max_results, 10)),
            "search_depth": "basic",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "query": query,
        "results": [
            {"title": item.get("title"), "url": item.get("url"), "content": item.get("content")}
            for item in payload.get("results", [])
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

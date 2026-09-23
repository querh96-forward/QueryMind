from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError

from app.config import get_settings
from app.northwind import METRIC_DEFINITIONS, NorthwindSource
from app.repository import Repository
from app.tool_errors import classify_tool_error

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception:  # pragma: no cover
    ClientSession = None
    streamable_http_client = None


class MCPWebSearchClient:
    """通过MCP调用外部联网搜索服务。"""

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or get_settings().web_mcp_url).strip()

    @property
    def available(self) -> bool:
        return bool(self.url and ClientSession and streamable_http_client)

    async def _search_async(self, query: str, max_results: int) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError("外部联网搜索MCP尚未配置")
        async with streamable_http_client(self.url) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(
                    "web_search",
                    {"query": query, "max_results": max_results},
                )
                structured = (
                    getattr(response, "structuredContent", None)
                    or getattr(response, "structured_content", None)
                )
                if structured:
                    return dict(structured)
                texts = []
                for item in getattr(response, "content", []) or []:
                    text = getattr(item, "text", None)
                    if text:
                        texts.append(text)
                return {"query": query, "results": texts}

    def search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        return asyncio.run(self._search_async(query, max_results))


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    source: str = "local"

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """One registry unifies local Python tools and external MCP tools."""

    def __init__(self, source: NorthwindSource, mcp_search: MCPWebSearchClient | None = None) -> None:
        self.source = source
        self.mcp_search = mcp_search or MCPWebSearchClient()
        self._tools: dict[str, ToolDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(ToolDefinition(
            name="inspect_schema",
            description="查看Northwind贸易经营数据库的表、视图和字段。生成SQL前可调用。",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=lambda _: {"schema": self.source.schema(compact=False)},
        ))
        self.register(ToolDefinition(
            name="query_database",
            description=(
                "执行一条只读PostgreSQL SELECT查询，分析订单、客户、商品、供应商、销售员工、物流和库存。"
                "禁止写入语句。销售额优先使用v_order_line_sales或v_order_summary中的sales_amount。"
            ),
            parameters={
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "只读SELECT SQL"}},
                "required": ["sql"],
                "additionalProperties": False,
            },
            handler=lambda args: self.source.query(str(args["sql"])),
        ))
        self.register(ToolDefinition(
            name="get_metric_definition",
            description="查询销售额、客单价、按期发货率、运费率、补货风险等业务指标口径。",
            parameters={
                "type": "object",
                "properties": {"metric_name": {"type": "string"}},
                "required": ["metric_name"],
                "additionalProperties": False,
            },
            handler=self._metric,
        ))
        if self.mcp_search.available:
            self.register(ToolDefinition(
                name="web_search",
                description="通过外部MCP服务器联网检索公开信息。仅在问题需要数据库外的最新信息时调用。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=lambda args: self.mcp_search.search(str(args["query"]), int(args.get("max_results", 5))),
                source="mcp",
            ))

    def _metric(self, args: dict[str, Any]) -> dict[str, Any]:
        name = str(args.get("metric_name", ""))
        matches = [item for item in METRIC_DEFINITIONS if name in item["name"] or item["name"] in name]
        return {"matches": matches or METRIC_DEFINITIONS}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(self, run_id: str, name: str, arguments: dict[str, Any], repo: Repository) -> dict[str, Any]:
        tool = self._tools.get(name)
        if not tool:
            result = {"ok": False, "error": f"未知工具：{name}", "error_type": "unknown_tool",
                      "retryable": True, "recovery_hint": "从提供的工具列表中选择工具。"}
            repo.add_tool_call(run_id, name, arguments, result, False, 0)
            return result
        started = time.perf_counter()
        try:
            Draft202012Validator(tool.parameters).validate(arguments)
            payload = tool.handler(arguments)
            result = {"ok": True, "source": tool.source, "data": payload}
            success = True
        except ValidationError as exc:
            result = {"ok": False, "source": tool.source, "error": exc.message,
                      "error_type": "invalid_arguments", "retryable": True,
                      "recovery_hint": "按工具的 JSON Schema 修正参数及类型。"}
            success = False
        except Exception as exc:
            result = {"ok": False, "source": tool.source, "error": str(exc), **classify_tool_error(exc)}
            success = False
        latency = round((time.perf_counter() - started) * 1000, 2)
        repo.add_tool_call(run_id, name, arguments, result, success, latency)
        return result

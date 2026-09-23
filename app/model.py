from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config import get_settings
from app.context import estimate_tokens

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


TokenCallback = Callable[[str], None]


@dataclass
class FunctionCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelDecision:
    content: str = ""
    tool_calls: list[FunctionCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class BaseModelGateway:
    def invoke(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: TokenCallback | None = None,
    ) -> ModelDecision:
        raise NotImplementedError


class QwenFunctionCallingGateway(BaseModelGateway):
    """Qwen Function Calling through DashScope's OpenAI-compatible API.

    LangGraph handles orchestration. This gateway only converts QueryMind's
    messages/tools to the OpenAI-compatible format and streams answer text
    back through ``on_token``. Tool-call arguments are accumulated from the
    streamed deltas before they are handed to the Tool Registry.
    """

    def __init__(self) -> None:
        settings = get_settings()
        if OpenAI is None:
            raise RuntimeError("请先安装openai依赖")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            timeout=min(60, settings.max_runtime_seconds),
            max_retries=0,
        )

    @staticmethod
    def _messages(system_prompt: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in messages:
            role = item.get("role")
            if role == "user":
                result.append({"role": "user", "content": item.get("content", "")})
            elif role == "assistant":
                message: dict[str, Any] = {"role": "assistant", "content": item.get("content", "")}
                calls = item.get("tool_calls", [])
                if calls:
                    message["tool_calls"] = [
                        {
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": call.get("name"),
                                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for call in calls
                    ]
                result.append(message)
            elif role == "tool":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("tool_call_id", "tool"),
                        "content": json.dumps(item.get("content", {}), ensure_ascii=False, default=str),
                    }
                )
        return result

    def invoke(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: TokenCallback | None = None,
    ) -> ModelDecision:
        api_messages = self._messages(system_prompt, messages)
        if on_token is None:
            response = self.client.chat.completions.create(
                model=self.settings.llm_model,
                messages=api_messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=0,
            )
            message = response.choices[0].message
            calls = []
            for item in message.tool_calls or []:
                try:
                    arguments = json.loads(item.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                calls.append(FunctionCall(id=item.id, name=item.function.name, arguments=arguments))
            usage = response.usage
            return ModelDecision(
                content=str(message.content or ""),
                tool_calls=calls,
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )

        stream_kwargs = {
            "model": self.settings.llm_model,
            "messages": api_messages,
            "tools": tools or None,
            "tool_choice": "auto" if tools else None,
            "temperature": 0,
            "stream": True,
        }
        try:
            stream = self.client.chat.completions.create(
                **stream_kwargs,
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            # Some OpenAI-compatible gateways do not implement stream_options.
            # Retry only for a clear 400 response; network/auth errors must surface.
            if getattr(exc, "status_code", None) != 400:
                raise
            stream = self.client.chat.completions.create(**stream_kwargs)
        content_parts: list[str] = []
        call_buffers: dict[int, dict[str, str]] = {}
        input_tokens = 0
        output_tokens = 0

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or input_tokens)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or output_tokens)
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            if content:
                content_parts.append(content)
                on_token(content)

            for item in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(item, "index", 0) or 0)
                buffer = call_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
                item_id = getattr(item, "id", None)
                if item_id:
                    buffer["id"] = item_id
                function = getattr(item, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    if name:
                        buffer["name"] += name
                    if arguments:
                        buffer["arguments"] += arguments

        calls: list[FunctionCall] = []
        for index in sorted(call_buffers):
            item = call_buffers[index]
            try:
                arguments = json.loads(item["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            calls.append(
                FunctionCall(
                    id=item["id"] or str(uuid.uuid4()),
                    name=item["name"],
                    arguments=arguments,
                )
            )

        content = "".join(content_parts)
        if not input_tokens:
            input_tokens = estimate_tokens(system_prompt + json.dumps(messages, ensure_ascii=False, default=str))
        if not output_tokens:
            output_tokens = estimate_tokens(content)
        return ModelDecision(
            content=content,
            tool_calls=calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class MockFunctionCallingGateway(BaseModelGateway):
    """Offline deterministic model for tests and first-time startup."""

    @staticmethod
    def _emit(content: str, on_token: TokenCallback | None) -> None:
        if not on_token:
            return
        for index in range(0, len(content), 8):
            on_token(content[index : index + 8])

    def invoke(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: TokenCallback | None = None,
    ) -> ModelDecision:
        last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        input_tokens = estimate_tokens(system_prompt + json.dumps(messages, ensure_ascii=False, default=str))

        if tool_messages:
            payload = tool_messages[-1].get("content", {})
            content = self._summarize(last_user, payload)
            self._emit(content, on_token)
            return ModelDecision(content=content, input_tokens=input_tokens, output_tokens=estimate_tokens(content))

        if any(word in last_user for word in ["你是谁", "介绍一下", "能做什么"]):
            content = "我是 QueryMind，可通过自然语言查询 Northwind 贸易经营数据，生成业务结论、表格和图表，并记录 Token 与运行指标。"
            self._emit(content, on_token)
            return ModelDecision(content=content, input_tokens=input_tokens, output_tokens=estimate_tokens(content))

        if any(word in last_user for word in ["最新", "联网", "网上", "新闻"]):
            names = {item["function"]["name"] for item in tools}
            if "web_search" in names:
                return ModelDecision(
                    tool_calls=[FunctionCall(str(uuid.uuid4()), "web_search", {"query": last_user, "max_results": 5})],
                    input_tokens=input_tokens,
                )
            content = "这个问题需要联网搜索，但当前尚未启动外部Web Search MCP服务。"
            self._emit(content, on_token)
            return ModelDecision(content=content, input_tokens=input_tokens, output_tokens=estimate_tokens(content))

        sql = self.mock_sql(last_user)
        return ModelDecision(
            tool_calls=[FunctionCall(str(uuid.uuid4()), "query_database", {"sql": sql})],
            input_tokens=input_tokens,
        )

    @staticmethod
    def mock_sql(question: str) -> str:
        q = question.lower()
        if any(k in q for k in ["每月", "月度", "趋势"]):
            return "SELECT to_char(order_date, 'YYYY-MM') AS 月份, ROUND(SUM(sales_amount),2) AS 销售额, COUNT(*) AS 订单量 FROM v_order_summary GROUP BY 月份 ORDER BY 月份"
        if "客户" in q and any(k in q for k in ["最高", "最多", "排名", "贡献"]):
            return "SELECT company_name AS 客户, country AS 国家, sales_amount AS 销售额, order_count AS 订单量, avg_order_value AS 客单价 FROM v_customer_sales ORDER BY sales_amount DESC LIMIT 10"
        if any(k in q for k in ["商品", "产品"]) and any(k in q for k in ["最高", "最多", "排名", "top"]):
            return "SELECT product_name AS 商品, category_name AS 品类, sales_amount AS 销售额, sales_quantity AS 销量 FROM v_product_sales ORDER BY sales_amount DESC LIMIT 10"
        if "品类" in q:
            return "SELECT category_name AS 品类, ROUND(SUM(sales_amount),2) AS 销售额, SUM(quantity) AS 销量 FROM v_order_line_sales GROUP BY category_name ORDER BY 销售额 DESC"
        if "运费" in q:
            return "SELECT ship_country AS 国家, ROUND(SUM(freight),2) AS 运费, ROUND(SUM(sales_amount),2) AS 销售额, ROUND(SUM(freight)/SUM(sales_amount)*100,2) AS 运费率 FROM v_order_summary GROUP BY ship_country ORDER BY 运费率 DESC"
        if any(k in q for k in ["国家", "地区"]):
            return "SELECT ship_country AS 国家, ROUND(SUM(sales_amount),2) AS 销售额, COUNT(*) AS 订单量 FROM v_order_summary GROUP BY ship_country ORDER BY 销售额 DESC LIMIT 15"
        if any(k in q for k in ["库存", "补货", "缺货"]):
            return "SELECT product_name AS 商品, category_name AS 品类, units_in_stock AS 库存, units_on_order AS 在途, reorder_level AS 补货点, inventory_status AS 状态 FROM v_inventory_status WHERE inventory_status <> '库存正常' ORDER BY 库存"
        if any(k in q for k in ["员工", "销售员"]):
            return "SELECT employee_name AS 销售员工, ROUND(SUM(sales_amount),2) AS 销售额, COUNT(*) AS 订单量 FROM v_order_summary GROUP BY employee_name ORDER BY 销售额 DESC"
        if any(k in q for k in ["发货", "履约", "准时"]):
            return "SELECT ROUND(AVG(shipping_days),2) AS 平均发货天数, ROUND(AVG(on_time)*100,2) AS 按期发货率, COUNT(*) AS 订单量 FROM v_order_summary WHERE shipped_date IS NOT NULL"
        return "SELECT ROUND(SUM(sales_amount),2) AS 销售额, COUNT(*) AS 订单量, ROUND(AVG(sales_amount),2) AS 客单价, ROUND(AVG(on_time)*100,2) AS 按期发货率 FROM v_order_summary"

    @staticmethod
    def _summarize(question: str, tool_payload: dict[str, Any]) -> str:
        if not tool_payload.get("ok"):
            return f"本次分析未完成：{tool_payload.get('error', '工具调用失败')}"
        data = tool_payload.get("data", {})
        rows = data.get("rows", []) if isinstance(data, dict) else []
        if not rows:
            if "results" in data:
                return "已完成联网检索，请查看下方的数据依据。"
            return "查询已执行，但没有找到符合条件的数据。"
        first = rows[0]
        pieces = []
        for key, value in list(first.items())[:4]:
            if isinstance(value, (int, float)):
                pieces.append(f"{key}为{value:,.2f}" if isinstance(value, float) else f"{key}为{value:,}")
            else:
                pieces.append(f"{key}是{value}")
        return "查询完成。" + "，".join(pieces) + "。下方表格和图表展示了完整结果。"


def build_model_gateway() -> BaseModelGateway:
    settings = get_settings()
    if settings.real_llm_enabled:
        return QwenFunctionCallingGateway()
    return MockFunctionCallingGateway()

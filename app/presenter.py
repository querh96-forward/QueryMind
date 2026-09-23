from __future__ import annotations

from typing import Any


def _number_cards(row: dict[str, Any]) -> list[dict[str, Any]]:
    cards = []
    for key, value in row.items():
        if isinstance(value, (int, float)) and len(cards) < 4:
            cards.append({"label": key, "value": value})
    return cards


def _chart(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    first = rows[0]
    text_columns = [k for k, v in first.items() if isinstance(v, str)]
    number_columns = [k for k, v in first.items() if isinstance(v, (int, float))]
    if not text_columns or not number_columns:
        return None
    x_key = text_columns[0]
    y_key = number_columns[0]
    chart_type = "line" if any(word in x_key for word in ["月", "日期", "时间", "年"]) else "bar"
    return {
        "type": chart_type,
        "title": f"{y_key}分析",
        "x_key": x_key,
        "y_key": y_key,
        "x": [row.get(x_key) for row in rows[:30]],
        "y": [row.get(y_key) for row in rows[:30]],
    }


def build_business_answer(
    text: str,
    tool_result: dict[str, Any] | None,
    *,
    tokens: dict[str, Any],
    verification_passed: bool,
) -> dict[str, Any]:
    """Convert technical tool output into the stable business UI contract."""
    data = (tool_result or {}).get("data", {}) if (tool_result or {}).get("ok") else {}
    rows = data.get("rows", []) if isinstance(data, dict) else []
    if not rows and isinstance(data, dict) and data.get("results"):
        rows = [
            {
                "标题": item.get("title", ""),
                "链接": item.get("url", ""),
                "摘要": item.get("content", "") if isinstance(item, dict) else str(item),
            }
            for item in data.get("results", [])
        ]
    source_name = "联网公开信息（MCP）" if (tool_result or {}).get("source") == "mcp" else ("Northwind贸易经营数据" if rows else (tool_result or {}).get("source", "模型回答"))
    result: dict[str, Any] = {
        "summary": text,
        "cards": _number_cards(rows[0]) if rows else [],
        "table": {"columns": list(rows[0]) if rows else [], "rows": rows[:200]},
        "chart": _chart(rows),
        "evidence": {
            "source": source_name,
            "tables": _tables_from_sql(data.get("sql", "")) if isinstance(data, dict) else [],
            "sql": data.get("sql", "") if isinstance(data, dict) else "",
            "row_count": data.get("row_count", len(rows)) if isinstance(data, dict) else 0,
            "query_latency_ms": data.get("latency_ms", 0) if isinstance(data, dict) else 0,
            "verification_passed": verification_passed,
        },
        "tokens": tokens,
    }
    return result


def _tables_from_sql(sql: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\b(?:from|join)\s+[\"`\[]?([a-zA-Z_][\w]*)", sql, re.I)))

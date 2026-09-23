"""Stable recovery information for tools; database messages remain observations."""
from __future__ import annotations

from typing import Any


class SQLPolicyError(ValueError):
    """A query violates the read-only/access policy and must not be retried."""


class SQLInputError(ValueError):
    """A malformed query can be repaired without relaxing the access policy."""


def classify_tool_error(exc: Exception) -> dict[str, Any]:
    cause = getattr(exc, "orig", exc)
    sqlstate = getattr(cause, "sqlstate", None) or getattr(cause, "pgcode", None)
    if isinstance(exc, SQLPolicyError) or sqlstate in {"42501", "25006"}:
        kind, retryable, hint = "permission_denied", False, "停止执行并说明访问限制，不要尝试绕过权限。"
    elif sqlstate in {"42703", "42P01"}:
        kind, retryable, hint = "schema_error", True, "先调用 inspect_schema 核对表和字段，再提交修正后的 SQL。"
    elif isinstance(exc, SQLInputError) or sqlstate == "42601":
        kind, retryable, hint = "sql_syntax", True, "根据语法错误修正一条 PostgreSQL SELECT，不要重复原查询。"
    elif sqlstate == "57014" or isinstance(exc, TimeoutError):
        kind, retryable, hint = "query_timeout", True, "简化查询；如需改变用户指定的统计范围，应先说明限制，不要静默改变口径。"
    elif sqlstate and sqlstate.startswith("08"):
        kind, retryable, hint = "database_unavailable", True, "数据库连接异常，可在本次调用限额内重试，仍失败则说明原因。"
    else:
        kind, retryable, hint = "tool_error", True, "检查工具参数和错误信息；无法修复时说明失败原因。"
    return {"error_type": kind, "retryable": retryable, "recovery_hint": hint}

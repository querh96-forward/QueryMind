"""Deterministic result comparison; never ask the model to grade itself."""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any


def _equal(a: Any, b: Any, tolerance: Decimal) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    numeric = (int, float, Decimal)
    if isinstance(a, numeric) and isinstance(b, numeric):
        left, right = Decimal(str(a)), Decimal(str(b))
        return left.is_finite() and right.is_finite() and abs(left - right) <= tolerance
    return type(a) is type(b) and a == b


def compare_rows(
    actual: list[dict], expected: list[dict], *, ordered: bool = True,
    abs_tolerance: str = "0.005", column_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Declared aliases map actual -> reference columns; unordered matching keeps duplicates."""
    tolerance = Decimal(str(abs_tolerance))
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("abs_tolerance must be a finite nonnegative number")
    aliases = column_aliases or {}
    mapped = []
    for row in actual:
        renamed = {aliases.get(key, key): value for key, value in row.items()}
        if len(renamed) != len(row):
            return {"correct": False, "reason": "column_alias_collision"}
        mapped.append(renamed)
    if len(mapped) != len(expected):
        return {"correct": False, "reason": "row_count_mismatch"}

    def matches(a, b):
        return a.keys() == b.keys() and all(_equal(a[key], b[key], tolerance) for key in a)

    if ordered:
        correct = all(matches(a, b) for a, b in zip(mapped, expected))
        return {"correct": correct, "reason": "matched" if correct else "ordered_value_or_column_mismatch"}

    # Bipartite matching avoids greedy tolerance comparisons rejecting valid rows.
    edges = [[j for j, row in enumerate(expected) if matches(candidate, row)] for candidate in mapped]
    owners: dict[int, int] = {}

    def assign(index, visited):
        for target in edges[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in owners or assign(owners[target], visited):
                owners[target] = index
                return True
        return False

    correct = all(assign(i, set()) for i in range(len(mapped)))
    return {"correct": correct, "reason": "matched" if correct else "unordered_value_or_column_mismatch"}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return round(values[lo] + (values[hi] - values[lo]) * (index - lo), 2)


def summarize(results: list[dict]) -> dict[str, Any]:
    def rate(n, d):
        return round(n / d * 100, 2) if d else None

    sql_cases = [item for item in results if item["first_sql_success"] is not None]
    failed_first = [item for item in sql_cases if item["first_sql_success"] is False]
    latencies = [item["latency_ms"] for item in results]
    n = len(results)
    return {
        "case_count": n,
        "result_accuracy": rate(sum(item["correct"] for item in results), n),
        "first_sql_success_rate": rate(sum(item["first_sql_success"] is True for item in sql_cases), len(sql_cases)),
        "sql_attempted_case_count": len(sql_cases),
        "recovery_eligible_count": len(failed_first),
        "recovery_result_accuracy": rate(sum(item["correct"] for item in failed_first), len(failed_first)),
        "timeout_count": sum(item["status"] == "eval_timeout" for item in results),
        "avg_latency_ms": round(sum(latencies) / n, 2) if n else 0,
        "p95_latency_ms": percentile(latencies, .95),
        "total_tokens": sum(item["total_tokens"] for item in results),
        "avg_tokens": round(sum(item["total_tokens"] for item in results) / n, 2) if n else 0,
        "avg_tool_calls": round(sum(item["tool_call_count"] for item in results) / n, 2) if n else 0,
    }

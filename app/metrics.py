from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any
from pathlib import Path
import json

from sqlalchemy import func, select

from app.db import EvalRun, Run, SessionLocal, ToolCall, UserFeedback


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return round(values[lower], 2)
    return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 2)


def system_metrics(days: int = 30) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with SessionLocal() as db:
        runs = list(db.scalars(select(Run).where(Run.created_at >= since).order_by(Run.created_at)))
        terminal = [r for r in runs if r.status in {"completed", "completed_with_warning", "failed", "canceled"}]
        completed = [r for r in terminal if r.status in {"completed", "completed_with_warning"}]
        latencies = [r.latency_ms for r in completed if r.latency_ms > 0]
        sql_runs = [r for r in terminal if r.first_sql_success is not None]
        feedback = list(db.scalars(select(UserFeedback).where(UserFeedback.created_at >= since)))
        latest_eval = db.scalar(select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1))

        def ratio(a: int, b: int) -> float:
            return round(a / b * 100, 2) if b else 0.0

        daily: dict[str, dict[str, float]] = {}
        for run in runs:
            day = run.created_at.date().isoformat()
            item = daily.setdefault(day, {"runs": 0, "tokens": 0, "cost": 0.0})
            item["runs"] += 1
            item["tokens"] += run.input_tokens + run.output_tokens + run.embedding_tokens
            item["cost"] += run.estimated_cost

        eval_metrics = latest_eval.metrics if latest_eval else None
        eval_name = latest_eval.dataset_name if latest_eval else None
        eval_cases = latest_eval.case_count if latest_eval else 0
        if latest_eval is None:
            report_path = Path(__file__).resolve().parents[1] / "evals" / "latest_report.json"
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    eval_metrics = report.get("metrics")
                    eval_name = report.get("dataset")
                    eval_cases = int(report.get("case_count", 0))
                except Exception:
                    pass

        metrics = {
            "period_days": days,
            "sample_count": len(terminal),
            "request_success_rate": ratio(len(completed), len(terminal)),
            "task_completion_rate": ratio(sum(1 for r in completed if r.verification_passed), len(terminal)),
            "verification_pass_rate": ratio(sum(1 for r in completed if r.verification_passed), len(completed)),
            "first_sql_success_rate": ratio(sum(1 for r in sql_runs if r.first_sql_success), len(sql_runs)),
            "p50_latency_ms": round(median(latencies), 2) if latencies else 0,
            "p95_latency_ms": percentile(latencies, 0.95),
            "avg_loops": round(sum(r.loop_count for r in completed) / len(completed), 2) if completed else 0,
            "avg_tool_calls": round(sum(r.tool_call_count for r in completed) / len(completed), 2) if completed else 0,
            "total_tokens": sum(r.input_tokens + r.output_tokens + r.embedding_tokens for r in terminal),
            "estimated_cost": round(sum(r.estimated_cost for r in terminal), 6),
            "satisfaction_rate": ratio(sum(1 for f in feedback if f.helpful), len(feedback)),
            "feedback_count": len(feedback),
            "daily": [{"date": k, **v, "cost": round(v["cost"], 6)} for k, v in sorted(daily.items())],
            "offline_evaluation": eval_metrics,
            "offline_evaluation_name": eval_name,
            "offline_evaluation_cases": eval_cases,
        }
        return metrics

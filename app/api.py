from __future__ import annotations

import asyncio
import csv
import io
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db import database_health
from app.metrics import system_metrics
from app.northwind import METRIC_DEFINITIONS
from app.service import AnalysisService


router = APIRouter(prefix="/api/v1")
service = AnalysisService()
settings = get_settings()
TERMINAL = {"completed", "completed_with_warning", "failed", "canceled"}


class ThreadCreate(BaseModel):
    title: str = "新分析"


class AnalysisCreate(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    token_budget: int | None = Field(default=None, ge=2000, le=100000)


class FeedbackCreate(BaseModel):
    helpful: bool
    comment: str = Field(default="", max_length=1000)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _thread(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "title": item.title,
        "summary": item.summary,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


def _message(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "role": item.role,
        "content": item.content,
        "run_id": item.run_id,
        "payload": item.payload,
        "created_at": _iso(item.created_at),
    }


def _run(item) -> dict[str, Any]:
    return {
        "id": item.id,
        "thread_id": item.thread_id,
        "question": item.question,
        "status": item.status,
        "answer": item.answer,
        "error": item.error,
        "verification_passed": item.verification_passed,
        "first_sql_success": item.first_sql_success,
        "loop_count": item.loop_count,
        "tool_call_count": item.tool_call_count,
        "token_budget": item.token_budget,
        "input_tokens": item.input_tokens,
        "output_tokens": item.output_tokens,
        "embedding_tokens": item.embedding_tokens,
        "estimated_cost": item.estimated_cost,
        "latency_ms": item.latency_ms,
        "created_at": _iso(item.created_at),
        "updated_at": _iso(item.updated_at),
    }


@router.get("/health")
def health() -> dict[str, Any]:
    queue = service.queue_status()
    database = database_health()
    row_counts = service.source.row_counts()
    ok = bool(database.get("ok") and database.get("pgvector") and queue.get("worker_online") and sum(row_counts.values()) == 3204)
    return {
        "status": "ok" if ok else "degraded",
        "app": settings.app_name,
        "version": settings.app_version,
        "runtime": service.graph.runtime_name,
        "llm": settings.llm_model if settings.real_llm_enabled else "离线Mock",
        "postgresql": database,
        "redis_queue": queue,
        "northwind_rows": sum(row_counts.values()),
        "mcp_web_search": bool(settings.web_mcp_url),
    }


@router.get("/public-config")
def public_config() -> dict[str, Any]:
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "default_token_budget": settings.default_token_budget,
        "max_agent_loops": settings.max_agent_loops,
        "data_source": "Northwind贸易经营数据",
        "real_llm_enabled": settings.real_llm_enabled,
    }


@router.post("/threads")
def create_thread(body: ThreadCreate) -> dict[str, Any]:
    return _thread(service.repo.create_thread(title=body.title.strip() or "新分析"))


@router.get("/threads")
def list_threads() -> list[dict[str, Any]]:
    return [_thread(item) for item in service.repo.list_threads()]


@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, bool]:
    service.repo.delete_thread(thread_id)
    return {"ok": True}


@router.get("/threads/{thread_id}/messages")
def list_messages(thread_id: str) -> list[dict[str, Any]]:
    try:
        service.repo.get_thread(thread_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return [_message(item) for item in service.repo.list_messages(thread_id)]


@router.post("/threads/{thread_id}/analyses")
def create_analysis(thread_id: str, body: AnalysisCreate) -> dict[str, str]:
    try:
        service.repo.get_thread(thread_id)
        run_id = service.create_analysis(
            thread_id=thread_id,
            question=body.question,
            token_budget=body.token_budget,
        )
        return {"run_id": run_id}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return _run(service.repo.get_run(run_id))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, bool]:
    try:
        service.cancel_run(run_id)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/runs/{run_id}/feedback")
def save_feedback(run_id: str, body: FeedbackCreate) -> dict[str, bool]:
    try:
        service.repo.get_run(run_id)
        service.repo.save_feedback(run_id, body.helpful, body.comment)
        return {"ok": True}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str):
    try:
        service.repo.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    async def generate():
        last_stream_id = "0-0"
        while True:
            stream_batches = await asyncio.to_thread(service.events.read, run_id, last_stream_id, 3000)
            emitted = False
            for _, entries in stream_batches:
                for stream_id, fields in entries:
                    last_stream_id = stream_id
                    emitted = True
                    event_id = fields.get("event_id") or stream_id
                    payload = {
                        "id": event_id,
                        "type": fields.get("event_type", "message"),
                        "data": json.loads(fields.get("data", "{}")),
                        "created_at": fields.get("created_at"),
                    }
                    yield f"id: {stream_id}\nevent: {payload['type']}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
            run = service.repo.get_run(run_id)
            if run.status in TERMINAL and not emitted:
                yield f"event: done\ndata: {json.dumps({'status': run.status}, ensure_ascii=False)}\n\n"
                break
            if not emitted:
                yield ": keep-alive\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@router.get("/runs/{run_id}/export.csv")
def export_run_csv(run_id: str):
    try:
        run = service.repo.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    rows = (run.answer or {}).get("table", {}).get("rows", [])
    if not rows:
        raise HTTPException(404, "该回答没有可导出的明细数据")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    content = "\ufeff" + buffer.getvalue()
    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="querymind-{run_id[:8]}.csv"'},
    )


@router.get("/dashboard/business")
def business_dashboard() -> dict[str, Any]:
    return service.source.dashboard()


@router.get("/dashboard/system")
def system_dashboard(days: int = 30) -> dict[str, Any]:
    return system_metrics(max(1, min(days, 365)))


@router.get("/data-source")
def data_source() -> dict[str, Any]:
    counts = service.source.row_counts()
    return {
        "name": "Northwind贸易经营数据",
        "type": "PostgreSQL只读业务Schema",
        "status": "正常",
        "tables": service.source.schema(compact=True),
        "row_counts": counts,
        "total_rows": sum(counts.values()),
    }


@router.get("/metric-definitions")
def metric_definitions() -> list[dict[str, str]]:
    return METRIC_DEFINITIONS


@router.get("/memories")
def list_memories() -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "key": item.key,
            "value": item.value,
            "memory_type": item.memory_type,
            "confidence": item.confidence,
            "updated_at": _iso(item.updated_at),
        }
        for item in service.repo.list_memories()
    ]


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str) -> dict[str, bool]:
    service.repo.delete_memory(memory_id)
    return {"ok": True}

from __future__ import annotations

import time
from typing import Any

import redis

from app.config import get_settings
from app.context import ContextBuilder
from app.graph import AnalysisGraph
from app.model import build_model_gateway
from app.northwind import NorthwindSource
from app.prompts import PromptRegistry
from app.rag import HybridRetriever
from app.repository import Repository
from app.runtime import EventBus
from app.tools import ToolRegistry


class AnalysisService:
    """Shared application facade used by API, Worker and MCP Server."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.repo = Repository()
        self.redis = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.redis.ping()
        self.source = NorthwindSource()
        self.retriever = HybridRetriever()
        self.prompts = PromptRegistry()
        self.tools = ToolRegistry(self.source)
        self.events = EventBus(self.repo)
        self.context_builder = ContextBuilder(self.repo, self.source, self.retriever)
        self.graph = AnalysisGraph(
            self.repo,
            self.context_builder,
            self.prompts,
            build_model_gateway(),
            self.tools,
            self.events,
        )

    def create_analysis(
        self,
        *,
        thread_id: str,
        question: str,
        user_id: str = "guest",
        token_budget: int | None = None,
    ) -> str:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        budget = max(2_000, min(token_budget or self.settings.default_token_budget, 100_000))
        user_message = self.repo.add_message(thread_id, "user", question)
        run = self.repo.create_run(thread_id, user_id, question, budget)
        self.repo.update_run(run.id, state={"user_message_id": user_message.id})
        self.events.publish(run.id, "queued", {"message": "问题已进入Redis任务队列"})
        thread = self.repo.get_thread(thread_id)
        if thread.title == "新分析":
            self.repo.update_thread(thread_id, title=question[:28])
        self.redis.rpush(self.settings.redis_queue_name, run.id)
        return run.id

    def execute_run(self, run_id: str) -> None:
        started = time.perf_counter()
        run = self.repo.get_run(run_id)
        if run.status in {"completed", "completed_with_warning", "failed", "canceled"}:
            return
        self.repo.update_run(run_id, status="running")
        self.events.publish(run_id, "started", {"message": "开始分析问题", "runtime": self.graph.runtime_name})
        try:
            state = self.graph.invoke({
                "run_id": run.id,
                "thread_id": run.thread_id,
                "user_id": run.user_id,
                "question": run.question,
                "token_budget": run.token_budget,
            })
            if self.repo.get_run(run_id).status == "canceled":
                return
            payload = state.get("answer_payload", {})
            token_data = payload.get("tokens", {})
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            payload["process"] = self._process_summary(
                run_id,
                latency_ms=latency_ms,
                loop_count=int(state.get("loop_count", 0)),
                tool_call_count=int(state.get("tool_call_count", 0)),
            )
            self.repo.update_run(
                run_id,
                status="completed" if state.get("verification_passed") else "completed_with_warning",
                answer=payload,
                state={
                    "runtime": self.graph.runtime_name,
                    "loop_count": state.get("loop_count", 0),
                    "tool_call_count": state.get("tool_call_count", 0),
                    "stop_reason": state.get("stop_reason", ""),
                    "sql_failure_count": state.get("sql_failure_count", 0),
                },
                verification_passed=bool(state.get("verification_passed")),
                first_sql_success=state.get("first_sql_success"),
                loop_count=int(state.get("loop_count", 0)),
                tool_call_count=int(state.get("tool_call_count", 0)),
                input_tokens=int(token_data.get("input", 0)),
                output_tokens=int(token_data.get("output", 0)),
                embedding_tokens=int(token_data.get("embedding", 0)),
                estimated_cost=float(token_data.get("estimated_cost", 0)),
                latency_ms=latency_ms,
            )
            self.repo.add_message(
                run.thread_id,
                "assistant",
                payload.get("summary", "分析完成"),
                run_id=run_id,
                payload=payload,
            )
            self._update_thread_summary(run.thread_id)
        except Exception as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.repo.update_run(run_id, status="failed", error=str(exc), latency_ms=latency_ms)
            self.events.publish(run_id, "failed", {"message": "分析失败", "error": str(exc)})

    def _process_summary(
        self,
        run_id: str,
        *,
        latency_ms: float,
        loop_count: int,
        tool_call_count: int,
    ) -> dict[str, Any]:
        """Build a business-friendly process summary without exposing hidden reasoning."""
        visible_types = {
            "started",
            "context_building",
            "context_ready",
            "thinking",
            "tool_running",
            "tool_finished",
            "verifying",
            "replanning",
            "verified",
            "completed",
            "stopped",
        }
        steps: list[dict[str, str]] = []
        last_message = ""
        for event in self.repo.list_events(run_id):
            if event.event_type not in visible_types:
                continue
            message = str((event.data or {}).get("message", "")).strip()
            if not message or message == last_message:
                continue
            steps.append({"type": event.event_type, "message": message})
            last_message = message
        return {
            "steps": steps[-10:],
            "duration_ms": latency_ms,
            "loop_count": loop_count,
            "tool_call_count": tool_call_count,
        }

    def cancel_run(self, run_id: str) -> None:
        run = self.repo.get_run(run_id)
        if run.status in {"completed", "completed_with_warning", "failed", "canceled"}:
            return
        self.repo.update_run(run_id, status="canceled", error="用户取消了任务")
        self.events.publish(run_id, "canceled", {"message": "任务已取消"})

    def _update_thread_summary(self, thread_id: str) -> None:
        messages = self.repo.list_messages(thread_id, limit=12)
        if len(messages) < 8:
            return
        summary = "；".join(
            f"{'用户' if message.role == 'user' else '系统'}：{message.content[:80]}"
            for message in messages[-6:]
        )
        self.repo.update_thread(thread_id, summary=summary[:1000])

    def queue_status(self) -> dict[str, Any]:
        heartbeat = self.redis.get(self.settings.redis_worker_heartbeat_key)
        return {
            "enabled": True,
            "mode": "Redis队列 + 独立Worker",
            "pending": self.redis.llen(self.settings.redis_queue_name),
            "worker_online": bool(heartbeat),
            "worker_heartbeat": heartbeat,
        }

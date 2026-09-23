from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from app.db import (
    ContextSnapshot,
    EvalRun,
    Message,
    PromptSnapshot,
    Run,
    RunEvent,
    SessionLocal,
    Thread,
    ToolCall,
    UserFeedback,
    UserMemory,
    utcnow,
)


class Repository:
    """集中管理QueryMind自身数据的读写。"""

    def create_thread(self, user_id: str = "guest", title: str = "新分析") -> Thread:
        with SessionLocal() as db:
            thread = Thread(user_id=user_id, title=title)
            db.add(thread)
            db.commit()
            db.refresh(thread)
            return thread

    def list_threads(self, user_id: str = "guest") -> list[Thread]:
        with SessionLocal() as db:
            statement = (
                select(Thread)
                .where(Thread.user_id == user_id)
                .order_by(Thread.updated_at.desc())
            )
            return list(db.scalars(statement))

    def get_thread(self, thread_id: str) -> Thread:
        with SessionLocal() as db:
            thread = db.get(Thread, thread_id)
            if not thread:
                raise KeyError("对话不存在")
            return thread

    def update_thread(
        self,
        thread_id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> None:
        with SessionLocal() as db:
            thread = db.get(Thread, thread_id)
            if not thread:
                raise KeyError("对话不存在")
            if title is not None:
                thread.title = title
            if summary is not None:
                thread.summary = summary
            thread.updated_at = utcnow()
            db.commit()

    def delete_thread(self, thread_id: str) -> None:
        with SessionLocal() as db:
            thread = db.get(Thread, thread_id)
            if thread:
                db.delete(thread)
                db.commit()

    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        *,
        run_id: str | None = None,
        payload: dict | None = None,
    ) -> Message:
        with SessionLocal() as db:
            message = Message(
                thread_id=thread_id,
                role=role,
                content=content,
                run_id=run_id,
                payload=payload or {},
            )
            db.add(message)
            thread = db.get(Thread, thread_id)
            if thread:
                thread.updated_at = utcnow()
            db.commit()
            db.refresh(message)
            return message

    def list_messages(self, thread_id: str, limit: int = 100) -> list[Message]:
        with SessionLocal() as db:
            statement = (
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            return list(reversed(list(db.scalars(statement))))

    def create_run(
        self,
        thread_id: str,
        user_id: str,
        question: str,
        token_budget: int,
    ) -> Run:
        with SessionLocal() as db:
            run = Run(
                thread_id=thread_id,
                user_id=user_id,
                question=question,
                token_budget=token_budget,
            )
            db.add(run)
            db.commit()
            db.refresh(run)
            return run

    def get_run(self, run_id: str) -> Run:
        with SessionLocal() as db:
            run = db.get(Run, run_id)
            if not run:
                raise KeyError("运行记录不存在")
            return run

    def update_run(self, run_id: str, **fields: Any) -> Run:
        with SessionLocal() as db:
            run = db.get(Run, run_id)
            if not run:
                raise KeyError("运行记录不存在")
            for key, value in fields.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            run.updated_at = utcnow()
            db.commit()
            db.refresh(run)
            return run

    def add_event(
        self,
        run_id: str,
        event_type: str,
        data: dict | None = None,
    ) -> RunEvent:
        with SessionLocal() as db:
            event = RunEvent(run_id=run_id, event_type=event_type, data=data or {})
            db.add(event)
            db.commit()
            db.refresh(event)
            return event

    def list_events(self, run_id: str, after_id: int = 0) -> list[RunEvent]:
        with SessionLocal() as db:
            statement = (
                select(RunEvent)
                .where(RunEvent.run_id == run_id, RunEvent.id > after_id)
                .order_by(RunEvent.id)
            )
            return list(db.scalars(statement))

    def add_tool_call(
        self,
        run_id: str,
        tool_name: str,
        arguments: dict,
        result: dict,
        success: bool,
        latency_ms: float,
    ) -> ToolCall:
        with SessionLocal() as db:
            call = ToolCall(
                run_id=run_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                success=success,
                latency_ms=latency_ms,
            )
            db.add(call)
            db.commit()
            db.refresh(call)
            return call

    def list_tool_calls(self, run_id: str) -> list[ToolCall]:
        with SessionLocal() as db:
            statement = (
                select(ToolCall)
                .where(ToolCall.run_id == run_id)
                .order_by(ToolCall.created_at)
            )
            return list(db.scalars(statement))

    def save_prompt_snapshot(
        self,
        run_id: str,
        name: str,
        version: str,
        prompt_hash: str,
        rendered: str,
    ) -> None:
        with SessionLocal() as db:
            db.add(
                PromptSnapshot(
                    run_id=run_id,
                    prompt_name=name,
                    prompt_version=version,
                    prompt_hash=prompt_hash,
                    rendered=rendered,
                )
            )
            db.commit()

    def save_context_snapshot(
        self,
        run_id: str,
        context: dict,
        estimated_tokens: int,
    ) -> None:
        with SessionLocal() as db:
            db.add(
                ContextSnapshot(
                    run_id=run_id,
                    context=context,
                    estimated_tokens=estimated_tokens,
                )
            )
            db.commit()

    def list_memories(self, user_id: str = "guest") -> list[UserMemory]:
        with SessionLocal() as db:
            statement = (
                select(UserMemory)
                .where(UserMemory.user_id == user_id)
                .order_by(UserMemory.updated_at.desc())
            )
            return list(db.scalars(statement))

    def upsert_memory(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        memory_type: str = "preference",
        confidence: float = 0.8,
        source_message_id: str | None = None,
    ) -> UserMemory:
        with SessionLocal() as db:
            statement = select(UserMemory).where(
                UserMemory.user_id == user_id,
                UserMemory.key == key,
            )
            memory = db.scalar(statement)
            if memory:
                memory.value = value
                memory.confidence = confidence
                memory.updated_at = utcnow()
                memory.source_message_id = source_message_id
            else:
                memory = UserMemory(
                    user_id=user_id,
                    key=key,
                    value=value,
                    memory_type=memory_type,
                    confidence=confidence,
                    source_message_id=source_message_id,
                )
                db.add(memory)
            db.commit()
            db.refresh(memory)
            return memory

    def delete_memory(self, memory_id: str) -> None:
        with SessionLocal() as db:
            memory = db.get(UserMemory, memory_id)
            if memory:
                db.delete(memory)
                db.commit()

    def save_feedback(
        self,
        run_id: str,
        helpful: bool,
        comment: str = "",
    ) -> UserFeedback:
        with SessionLocal() as db:
            feedback = db.scalar(
                select(UserFeedback).where(UserFeedback.run_id == run_id)
            )
            if feedback:
                feedback.helpful = helpful
                feedback.comment = comment
            else:
                feedback = UserFeedback(
                    run_id=run_id,
                    helpful=helpful,
                    comment=comment,
                )
                db.add(feedback)
            db.commit()
            db.refresh(feedback)
            return feedback

    def save_eval_run(
        self,
        dataset_name: str,
        model_name: str,
        metrics: dict,
        case_count: int,
    ) -> EvalRun:
        with SessionLocal() as db:
            result = EvalRun(
                dataset_name=dataset_name,
                model_name=model_name,
                metrics=metrics,
                case_count=case_count,
            )
            db.add(result)
            db.commit()
            db.refresh(result)
            return result

    def latest_eval_run(self) -> EvalRun | None:
        with SessionLocal() as db:
            statement = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(1)
            return db.scalar(statement)

    def list_recoverable_run_ids(self, minutes: int = 2) -> list[str]:
        """将排队任务和长时间无更新的运行任务重新放回队列。"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        with SessionLocal() as db:
            statement = select(Run).where(
                (Run.status == "queued")
                | (
                    Run.status.in_(["running", "verifying"])
                    & (Run.updated_at < cutoff)
                )
            )
            runs = list(db.scalars(statement))
            for run in runs:
                run.status = "queued"
                run.error = ""
                run.updated_at = utcnow()
            db.commit()
            return [run.id for run in runs]

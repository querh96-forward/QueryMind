from __future__ import annotations

import json
from typing import Any

import psycopg
import redis
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from app.config import get_settings
from app.repository import Repository


class CheckpointManager:
    """管理LangGraph的PostgreSQL检查点连接。"""

    def __init__(self) -> None:
        settings = get_settings()
        self.connection = psycopg.connect(
            settings.checkpoint_database_url,
            autocommit=True,
            prepare_threshold=0,
            row_factory=dict_row,
        )
        self.saver = PostgresSaver(self.connection)

    def setup(self) -> None:
        self.saver.setup()

    def close(self) -> None:
        self.connection.close()


class EventBus:
    """PostgreSQL保存审计事件，Redis Streams推送实时事件。"""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo
        self.settings = get_settings()
        self.client = redis.Redis.from_url(self.settings.redis_url, decode_responses=True)
        self.client.ping()

    def stream_name(self, run_id: str) -> str:
        return f"{self.settings.redis_stream_prefix}:{run_id}"

    def publish(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        durable: bool = True,
    ) -> None:
        payload = data or {}
        event_id = ""
        created_at = ""
        if durable:
            event = self.repo.add_event(run_id, event_type, payload)
            event_id = str(event.id)
            created_at = event.created_at.isoformat()
        self.client.xadd(
            self.stream_name(run_id),
            {
                "event_id": event_id,
                "event_type": event_type,
                "data": json.dumps(payload, ensure_ascii=False, default=str),
                "created_at": created_at,
            },
            maxlen=3000,
            approximate=True,
        )

    def read(self, run_id: str, last_stream_id: str = "0-0", block_ms: int = 5000):
        return self.client.xread(
            {self.stream_name(run_id): last_stream_id},
            count=200,
            block=block_ms,
        )

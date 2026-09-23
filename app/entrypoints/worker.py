from __future__ import annotations

import threading
import time

import redis

from app.config import get_settings
from app.service import AnalysisService


def start_heartbeat(client: redis.Redis, key: str) -> None:
    """Keep Docker and API informed even while one analysis is running."""
    def loop() -> None:
        while True:
            client.set(key, str(time.time()), ex=15)
            time.sleep(5)

    threading.Thread(target=loop, name="worker-heartbeat", daemon=True).start()


def main() -> None:
    settings = get_settings()
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    client.ping()
    start_heartbeat(client, settings.redis_worker_heartbeat_key)
    service = AnalysisService()

    # Requeue work that existed before a worker restart. Duplicate queue items
    # are harmless because execute_run ignores terminal runs.
    for run_id in service.repo.list_recoverable_run_ids():
        client.rpush(settings.redis_queue_name, run_id)

    print(f"QueryMind Worker已启动，队列：{settings.redis_queue_name}", flush=True)
    while True:
        item = client.blpop(settings.redis_queue_name, timeout=5)
        if not item:
            continue
        _, run_id = item
        try:
            service.execute_run(run_id)
        except Exception as exc:
            print(f"任务 {run_id} 执行失败：{exc}", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()

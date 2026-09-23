from __future__ import annotations

import time
import json

import psycopg
from psycopg import sql
from sqlalchemy import text

from app.config import get_settings
from app.db import engine, init_db
from app.northwind import SCHEMA_SQL, TABLES, VIEW_SQL, csv_rows
from app.rag import HybridRetriever
from app.runtime import CheckpointManager

settings = get_settings()


def wait_for_postgres(attempts: int = 60) -> None:
    for _ in range(attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("PostgreSQL在60秒内没有就绪")


def ensure_reader_role() -> None:
    # The reader role makes the Agent's business SQL physically read-only.
    with psycopg.connect(settings.checkpoint_database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'northwind_reader'")
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier("northwind_reader"), sql.Literal(settings.northwind_reader_password)
                ))
            else:
                cur.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                    sql.Identifier("northwind_reader"), sql.Literal(settings.northwind_reader_password)
                ))


def execute_script(script: str) -> None:
    statements = [part.strip() for part in script.split(";") if part.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def seed_table(table: str) -> int:
    rows = csv_rows(table)
    with engine.begin() as conn:
        exists = int(conn.execute(text(f"SELECT COUNT(*) FROM northwind.{table}")).scalar_one())
        if exists:
            return exists
        if not rows:
            return 0
        columns = list(rows[0])
        placeholders = ", ".join(f":{name}" for name in columns)
        names = ", ".join(columns)
        conn.execute(text(f"INSERT INTO northwind.{table} ({names}) VALUES ({placeholders})"), rows)
        return len(rows)


def grant_reader() -> None:
    with engine.begin() as conn:
        conn.execute(text("GRANT CONNECT ON DATABASE querymind TO northwind_reader"))
        conn.execute(text("GRANT USAGE ON SCHEMA northwind TO northwind_reader"))
        conn.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA northwind TO northwind_reader"))
        conn.execute(text("ALTER DEFAULT PRIVILEGES IN SCHEMA northwind GRANT SELECT ON TABLES TO northwind_reader"))


def main() -> None:
    print("[bootstrap] 等待PostgreSQL")
    wait_for_postgres()
    print("[bootstrap] 初始化系统表、pgvector和pg_trgm")
    init_db()
    ensure_reader_role()
    print("[bootstrap] 创建Northwind表")
    execute_script(SCHEMA_SQL)
    # Foreign-key-safe order.
    order = ["categories", "customers", "employees", "shippers", "suppliers", "products", "orders", "order_details"]
    for table in order:
        print(f"[bootstrap] {table}: {seed_table(table)} rows")
    execute_script(VIEW_SQL)
    grant_reader()
    print("[bootstrap] 初始化Hybrid RAG向量索引")
    retriever = HybridRetriever()
    print("[bootstrap] " + json.dumps(retriever.ensure_index(), ensure_ascii=False))
    print("[bootstrap] 初始化LangGraph PostgreSQL Checkpoint表")
    checkpoint = CheckpointManager()
    checkpoint.setup()
    checkpoint.close()
    print("[bootstrap] 完成")


if __name__ == "__main__":
    main()

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """QueryMind has one deployment shape: PostgreSQL + Redis + Worker."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "QueryMind：基于 LangGraph 的可治理企业数据智能分析平台"
    app_version: str = "6.2.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 6006

    # PostgreSQL is the only durable database. It stores conversations, runs,
    # memories, evaluations, RAG vectors and the Northwind business schema.
    database_url: str = "postgresql+psycopg://querymind:querymind@postgres:5432/querymind"
    business_database_url: str = "postgresql+psycopg://northwind_reader:northwind_reader@postgres:5432/querymind"
    northwind_reader_password: str = "northwind_reader"

    # Redis is always enabled: queue, worker heartbeat, event streams and cache.
    redis_url: str = "redis://redis:6379/0"
    redis_queue_name: str = "querymind:runs"
    redis_stream_prefix: str = "querymind:events"
    redis_worker_heartbeat_key: str = "querymind:worker:heartbeat"
    rag_cache_seconds: int = 300

    llm_model: str = "qwen-plus"
    dashscope_api_key: str = Field(default="", validation_alias=AliasChoices("LLM_API_KEY", "DASHSCOPE_API_KEY"))
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    mock_llm: bool = False

    embedding_provider: Literal["openai", "hash"] = "openai"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = Field(
        default=1024, ge=1, le=2000,
        validation_alias=AliasChoices("EMBEDDING_DIMENSIONS", "RAG_EMBEDDING_DIM"),
    )
    embedding_timeout_seconds: float = Field(default=30, gt=0)
    embedding_batch_size: int = Field(default=10, ge=1, le=10)

    default_token_budget: int = 20_000
    max_agent_loops: int = Field(default=5, ge=1, le=50)
    max_tool_calls: int = Field(default=8, ge=1, le=100)
    max_runtime_seconds: int = Field(default=90, gt=0)
    max_sql_repairs: int = Field(default=2, ge=0, le=10)
    input_cost_per_million: float = 0.8
    output_cost_per_million: float = 2.0
    embedding_cost_per_million: float = 0.1

    web_mcp_url: str = "http://web-mcp:8011/mcp"
    mcp_host: str = "0.0.0.0"
    querymind_mcp_port: int = 8010
    web_mcp_port: int = 8011
    tavily_api_key: str = ""

    rag_top_k: int = Field(default=5, ge=1, le=100)

    @property
    def rag_embedding_dim(self) -> int:
        return self.embedding_dimensions

    @property
    def checkpoint_database_url(self) -> str:
        """LangGraph PostgresSaver expects a normal psycopg URL."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    @property
    def real_llm_enabled(self) -> bool:
        return bool(self.dashscope_api_key.strip()) and not self.mock_llm


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from app.northwind import NorthwindSource
from app.rag import HybridRetriever
from app.repository import Repository


def estimate_tokens(text: str) -> int:
    # A transparent estimate used before the provider returns exact usage.
    chinese = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other = max(0, len(text) - chinese)
    return max(1, int(chinese / 1.5 + other / 4))


@dataclass
class ContextPack:
    question: str
    recent_messages: list[dict[str, str]]
    thread_summary: str
    user_memories: list[dict[str, Any]]
    schema: str
    knowledge: list[dict[str, Any]]
    estimated_tokens: int
    token_budget: int
    compressed: bool = False
    embedding_tokens: int = 0
    retrieval_cache_hit: bool = False
    retrieval_index_version: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_text(self) -> str:
        memory_text = "\n".join(f"- {m['key']}：{m['value']}" for m in self.user_memories) or "- 无"
        knowledge_text = "\n".join(f"- [{k['kind']}] {k['title']}：{k['content']}" for k in self.knowledge) or "- 无"
        conversation = "\n".join(f"{m['role']}：{m['content']}" for m in self.recent_messages) or "无"
        return f"""当前问题：{self.question}

最近对话：
{conversation}

对话摘要：{self.thread_summary or '无'}

用户偏好：
{memory_text}

相关数据结构：
{self.schema}

检索到的业务知识：
{knowledge_text}
""".strip()


class ContextBuilder:
    def __init__(self, repo: Repository, source: NorthwindSource, retriever: HybridRetriever) -> None:
        self.repo = repo
        self.source = source
        self.retriever = retriever

    def build(self, *, run_id: str, thread_id: str, user_id: str, question: str, token_budget: int) -> ContextPack:
        messages = self.repo.list_messages(thread_id, limit=10)
        recent = [{"role": m.role, "content": m.content[:1500]} for m in messages[-8:]]
        thread = self.repo.get_thread(thread_id)
        memories = [
            {"id": m.id, "key": m.key, "value": m.value, "confidence": m.confidence}
            for m in self.repo.list_memories(user_id)[:8]
        ]
        retrieval = self.retriever.retrieve_with_usage(question)
        hits = [hit.as_dict() for hit in retrieval.hits]
        schema = self.source.schema_text()
        draft = ContextPack(
            question=question, recent_messages=recent, thread_summary=thread.summary,
            user_memories=memories, schema=schema, knowledge=hits,
            estimated_tokens=0, token_budget=token_budget,
            embedding_tokens=retrieval.embedding_tokens,
            retrieval_cache_hit=retrieval.cache_hit,
            retrieval_index_version=retrieval.index_version,
        )
        estimated = estimate_tokens(draft.prompt_text())
        # Context gets at most 55% of total budget, leaving room for tool output
        # and the final answer.
        max_context = max(1200, int(token_budget * 0.55))
        if estimated > max_context:
            draft.recent_messages = draft.recent_messages[-4:]
            draft.knowledge = draft.knowledge[:3]
            draft.schema = "\n".join(draft.schema.splitlines()[:10])
            draft.compressed = True
            estimated = estimate_tokens(draft.prompt_text())
        draft.estimated_tokens = estimated
        self.repo.save_context_snapshot(run_id, draft.as_dict(), estimated)
        return draft


@dataclass
class MemoryCandidate:
    key: str
    value: str
    confidence: float = 0.9
    memory_type: str = "preference"


class MemoryExtractor:
    """只保存用户明确表达的长期偏好，避免把临时问题写入记忆。"""

    RULES = [
        (r"(?:以后|今后|默认)(?:都)?用([\u4e00-\u9fffA-Za-z]+图)", "chart_type", lambda m: m.group(1)),
        (r"金额(?:默认)?保留(\d+)位", "decimal_places", lambda m: m.group(1)),
        (r"(?:以后|默认)(?:先|优先)看(同比|环比|趋势|排名)", "analysis_focus", lambda m: m.group(1)),
        (r"(?:以后|默认)使用([^，。]+)(?:数据源|数据库)", "default_source", lambda m: m.group(1).strip()),
    ]

    def extract(self, text: str) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for pattern, key, getter in self.RULES:
            match = re.search(pattern, text)
            if match:
                candidates.append(MemoryCandidate(key=key, value=getter(match)))
        return candidates

    def persist(
        self,
        repo: Repository,
        user_id: str,
        text: str,
        source_message_id: str | None = None,
    ) -> list[MemoryCandidate]:
        candidates = self.extract(text)
        for item in candidates:
            repo.upsert_memory(
                user_id,
                item.key,
                item.value,
                confidence=item.confidence,
                memory_type=item.memory_type,
                source_message_id=source_message_id,
            )
        return candidates

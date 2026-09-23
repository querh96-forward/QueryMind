from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import redis
import yaml
from sqlalchemy import text

from app.config import ROOT, Settings, get_settings
from app.db import engine
from app.embeddings import EmbeddingProvider, hash_embedding, tokenize  # backward-compatible imports


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{item:.8f}" for item in vector) + "]"


@dataclass
class KnowledgeHit:
    id: str
    kind: str
    title: str
    content: str
    score: float
    source: str = "Northwind知识库"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RetrievalResult:
    hits: list[KnowledgeHit]
    embedding_tokens: int
    cache_hit: bool
    index_version: str


class HybridRetriever:
    """Versioned pg_trgm + semantic pgvector retrieval, combined using RRF."""

    def __init__(self, path: str | Path | None = None, *, settings: Settings | None = None,
                 provider: EmbeddingProvider | None = None, redis_client: Any = None) -> None:
        self.settings = settings or get_settings()
        self.path = Path(path or ROOT / "data" / "knowledge.yaml")
        self.documents: list[dict[str, Any]] = yaml.safe_load(self.path.read_text(encoding="utf-8")) or []
        if not self.documents or len({item["id"] for item in self.documents}) != len(self.documents):
            raise ValueError("Knowledge corpus must contain documents with unique IDs")
        canonical = [{"id": d["id"], "kind": d.get("kind", "knowledge"), "title": d["title"],
                      "content": d["content"], "keywords": d.get("keywords", [])} for d in self.documents]
        self.corpus_sha256 = hashlib.sha256(json.dumps(
            sorted(canonical, key=lambda d: d["id"]), ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        self.provider = provider or EmbeddingProvider(self.settings)
        self.index_version = hashlib.sha256(
            f"rag-v2|{self.provider.fingerprint}|{self.corpus_sha256}".encode()
        ).hexdigest()[:24]
        # Interpolated SQL identifiers are derived exclusively from a hex digest.
        self.table = f"rag_knowledge_{self.index_version}"
        self.redis = redis_client if redis_client is not None else redis.Redis.from_url(
            self.settings.redis_url, decode_responses=True,
        )

    def index_info(self) -> dict[str, Any]:
        return {**self.provider.identity, "index_version": self.index_version,
                "corpus_sha256": self.corpus_sha256, "document_count": len(self.documents),
                "table": self.table}

    def _index_exists(self, conn: Any) -> bool:
        return conn.execute(text("SELECT to_regclass(:table)"), {"table": self.table}).scalar() is not None

    def ensure_index(self) -> dict[str, Any]:
        """Build a complete immutable generation atomically; retain older generations."""
        with engine.connect() as conn:
            if self._index_exists(conn):
                return {**self.index_info(), "created": False, "embedding_tokens": 0}
        # API failures happen before DDL. No partial index becomes visible to readers.
        batch = self.provider.embed([
            f"{d['title']} {d['content']} {' '.join(d.get('keywords', []))}" for d in self.documents
        ])
        with engine.begin() as conn:
            conn.execute(text("SELECT pg_advisory_xact_lock(:key)"),
                         {"key": int(self.index_version[:15], 16)})
            if self._index_exists(conn):
                return {**self.index_info(), "created": False, "embedding_tokens": batch.tokens}
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text("""CREATE TABLE IF NOT EXISTS rag_index_versions (
                version TEXT PRIMARY KEY, metadata JSONB NOT NULL,
                embedding_tokens INTEGER NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"""))
            conn.execute(text(f"""CREATE TABLE {self.table} (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
                content TEXT NOT NULL, keywords JSONB NOT NULL,
                embedding vector({self.provider.dimensions}) NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"""))
            for item, vector in zip(self.documents, batch.vectors, strict=True):
                conn.execute(text(f"""INSERT INTO {self.table}(id, kind, title, content, keywords, embedding)
                    VALUES (:id, :kind, :title, :content, CAST(:keywords AS jsonb), CAST(:embedding AS vector))
                """), {"id": item["id"], "kind": item.get("kind", "knowledge"),
                       "title": item["title"], "content": item["content"],
                       "keywords": json.dumps(item.get("keywords", []), ensure_ascii=False),
                       "embedding": vector_literal(vector)})
            conn.execute(text(f"CREATE INDEX ON {self.table} USING gin ((title || ' ' || content) gin_trgm_ops)"))
            conn.execute(text(f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)"))
            conn.execute(text("""INSERT INTO rag_index_versions(version, metadata, embedding_tokens)
                VALUES (:version, CAST(:metadata AS jsonb), :tokens)"""),
                {"version": self.index_version, "metadata": json.dumps(self.index_info()), "tokens": batch.tokens})
        return {**self.index_info(), "created": True, "embedding_tokens": batch.tokens}

    def cache_key(self, query: str, top_k: int, kinds: set[str] | None, strategy: str) -> str:
        payload = json.dumps([query, top_k, sorted(kinds or []), strategy], ensure_ascii=False)
        return f"querymind:rag:v3:{self.index_version}:" + hashlib.sha256(payload.encode()).hexdigest()

    def retrieve(self, query: str, top_k: int | None = None, kinds: set[str] | None = None) -> list[KnowledgeHit]:
        return self.retrieve_with_usage(query, top_k, kinds).hits

    def retrieve_with_usage(self, query: str, top_k: int | None = None, kinds: set[str] | None = None,
                            *, strategy: Literal["hybrid", "vector", "keyword", "legacy_hybrid"] = "hybrid",
                            use_cache: bool = True) -> RetrievalResult:
        if strategy not in {"hybrid", "vector", "keyword", "legacy_hybrid"}:
            raise ValueError("Unknown retrieval strategy")
        top_k = top_k if top_k is not None else self.settings.rag_top_k
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return RetrievalResult([], 0, False, self.index_version)
        key = self.cache_key(query, top_k, kinds, strategy)
        if use_cache and (cached := self.redis.get(key)):
            return RetrievalResult([KnowledgeHit(**d) for d in json.loads(cached)], 0, True, self.index_version)
        with engine.connect() as conn:
            if not self._index_exists(conn):
                raise RuntimeError("RAG index does not match this model/corpus. Run python -m scripts.bootstrap first.")
        params: dict[str, Any] = {"query": query, "limit": max(20, top_k * 4)}
        embedding_tokens = 0
        if strategy != "keyword":
            batch = self.provider.embed([query])
            params["embedding"] = vector_literal(batch.vectors[0])
            embedding_tokens = batch.tokens
        kind_sql = ""
        if kinds:
            params["kinds"] = sorted(kinds)
            kind_sql = "WHERE kind = ANY(:kinds)"
        keyword_sql = kind_sql
        if strategy != "legacy_hybrid":
            # Zero similarity is no lexical evidence. Padding those rows lets ID order
            # manufacture RRF votes and overwhelm genuine semantic neighbours.
            keyword_sql += (" AND " if kind_sql else " WHERE ") + "similarity(title || ' ' || content, :query) > 0"
        with engine.connect() as conn:
            keyword_rows = [] if strategy == "vector" else conn.execute(text(f"""
                SELECT id, similarity(title || ' ' || content, :query) AS score
                FROM {self.table} {keyword_sql} ORDER BY score DESC, id LIMIT :limit
            """), params).mappings().all()
            vector_rows = [] if strategy == "keyword" else conn.execute(text(f"""
                SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM {self.table} {kind_sql}
                ORDER BY embedding <=> CAST(:embedding AS vector), id LIMIT :limit
            """), params).mappings().all()
            rrf: dict[str, float] = {}
            for rows in (keyword_rows, vector_rows):
                for rank, row in enumerate(rows, start=1):
                    rrf[row["id"]] = rrf.get(row["id"], 0.0) + 1 / (60 + rank)
            # Stable insertion-order tie handling preserves the original RRF baseline.
            selected = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
            rows = conn.execute(text(f"SELECT id, kind, title, content FROM {self.table} WHERE id = ANY(:ids)"),
                                {"ids": selected}).mappings().all() if selected else []
        by_id = {row["id"]: row for row in rows}
        hits = [KnowledgeHit(**dict(by_id[doc_id]), score=round(rrf[doc_id], 6)) for doc_id in selected]
        if use_cache and self.settings.rag_cache_seconds > 0:
            self.redis.setex(key, self.settings.rag_cache_seconds, json.dumps([h.as_dict() for h in hits], ensure_ascii=False))
        return RetrievalResult(hits, embedding_tokens, False, self.index_version)

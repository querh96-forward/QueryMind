"""Explicit embedding providers; production errors never fall back to hashing."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from app.config import Settings


def tokenize(value: str) -> set[str]:
    value = value.lower()
    tokens = set(re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", value))
    for block in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.add(block)
        tokens.update(block[index:index + 2] for index in range(max(1, len(block) - 1)))
    return {token for token in tokens if token.strip()}


def hash_embedding(value: str, dimension: int) -> list[float]:
    """Original baseline, available only through an explicit hash configuration."""
    vector = [0.0] * dimension
    for token in tokenize(value):
        number = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")
        vector[number % dimension] += 1.0 if number & 1 else -1.0
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    tokens: int = 0


class EmbeddingProvider:
    def __init__(self, settings: Settings, *, client: Any = None) -> None:
        self.settings = settings
        self.dimensions = settings.embedding_dimensions
        self.identity = {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model if settings.embedding_provider == "openai" else "blake2b-token-v1",
            "dimensions": self.dimensions,
            "endpoint_sha256": hashlib.sha256(settings.embedding_base_url.rstrip("/").encode()).hexdigest()
            if settings.embedding_provider == "openai" else None,
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.client = client
        if settings.embedding_provider == "openai" and client is None:
            if not settings.embedding_api_key.strip():
                raise ValueError("EMBEDDING_API_KEY is required for EMBEDDING_PROVIDER=openai")
            self.client = OpenAI(
                api_key=settings.embedding_api_key, base_url=settings.embedding_base_url,
                timeout=settings.embedding_timeout_seconds, max_retries=1,
            )

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch([])
        if any(not value.strip() for value in texts):
            raise ValueError("Embedding input must not be blank")
        if self.settings.embedding_provider == "hash":
            return EmbeddingBatch([hash_embedding(value, self.dimensions) for value in texts])
        vectors: list[list[float]] = []
        tokens = 0
        size = self.settings.embedding_batch_size
        for offset in range(0, len(texts), size):
            batch = texts[offset:offset + size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model, input=batch,
                dimensions=self.dimensions, encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("Embedding response indices/count do not match the request")
            for item in ordered:
                vector = item.embedding
                if (len(vector) != self.dimensions
                        or any(not math.isfinite(value) for value in vector)
                        or not any(value != 0 for value in vector)):
                    raise ValueError("Embedding response contains an invalid vector or dimension")
                vectors.append(vector)
            tokens += int(response.usage.total_tokens)
        return EmbeddingBatch(vectors, tokens)

"""Real PostgreSQL transaction/index tests; embedding calls remain offline."""
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from app.config import get_settings
from app.db import engine
from app.embeddings import EmbeddingBatch, EmbeddingProvider
from app.rag import HybridRetriever


class Cache:
    def __init__(self):
        self.values = {}
    def get(self, key):
        return self.values.get(key)
    def setex(self, key, ttl, value):
        self.values[key] = value


@pytest.fixture
def make_retriever(tmp_path):
    made = []
    def make(content='first', dimension=256):
        path = tmp_path / f'{len(made)}.yaml'
        path.write_text(f'- id: a\n  kind: metric\n  title: sales\n  content: {content} {tmp_path.name}\n'
                        '- id: b\n  kind: rule\n  title: freight\n  content: no duplicates\n')
        settings = get_settings().model_copy(update={'embedding_provider': 'hash', 'embedding_dimensions': dimension})
        provider = EmbeddingProvider(settings)
        original = provider.embed
        calls = []
        def embed(texts):
            calls.append(texts)
            result = original(texts)
            return EmbeddingBatch(result.vectors, 17)
        provider.embed = embed
        r = HybridRetriever(path, settings=settings, provider=provider, redis_client=Cache())
        made.append(r)
        return r, calls
    yield make
    with engine.begin() as conn:
        for r in made:
            conn.execute(text(f'DROP TABLE IF EXISTS {r.table}'))
            if conn.execute(text("SELECT to_regclass('rag_index_versions')")).scalar():
                conn.execute(text('DELETE FROM rag_index_versions WHERE version=:v'), {'v': r.index_version})


def test_index_reuse_cache_accounting_and_kind_filter(make_retriever):
    r, calls = make_retriever()
    assert r.ensure_index()['created'] is True
    assert r.ensure_index()['created'] is False
    assert len(calls) == 1
    cold = r.retrieve_with_usage('sales', kinds={'metric'})
    warm = r.retrieve_with_usage('sales', kinds={'metric'})
    assert [h.id for h in cold.hits] == ['a']
    assert cold.embedding_tokens == 17 and not cold.cache_hit
    assert warm.hits == cold.hits and warm.embedding_tokens == 0 and warm.cache_hit
    assert len(calls) == 2
    r.retrieve_with_usage('sales', kinds={'metric'}, strategy='keyword')
    assert len(calls) == 2


def test_new_dimension_and_corpus_keep_previous_generation(make_retriever):
    old, _ = make_retriever()
    changed, _ = make_retriever('changed', dimension=32)
    old.ensure_index()
    changed.ensure_index()
    assert old.index_version != changed.index_version
    with engine.connect() as conn:
        assert conn.execute(text(f'SELECT vector_dims(embedding) FROM {old.table} LIMIT 1')).scalar() == 256
        assert conn.execute(text(f'SELECT vector_dims(embedding) FROM {changed.table} LIMIT 1')).scalar() == 32
    assert old.retrieve('sales') and changed.retrieve('sales')


def test_partial_build_rolls_back_and_old_index_survives(make_retriever):
    old, _ = make_retriever()
    old.ensure_index()
    broken, _ = make_retriever('broken')
    broken.provider.embed = lambda texts: EmbeddingBatch([[1.0] * 256])  # too few vectors; rollback after one insert
    with pytest.raises(ValueError):
        broken.ensure_index()
    with engine.connect() as conn:
        assert not broken._index_exists(conn)
        assert old._index_exists(conn)
    assert old.retrieve('sales')


def test_provider_failure_and_missing_generation_are_explicit(make_retriever):
    r, _ = make_retriever('unavailable')
    def unavailable(texts):
        raise RuntimeError('provider unavailable')
    r.provider.embed = unavailable
    with pytest.raises(RuntimeError, match='provider unavailable'):
        r.ensure_index()
    with pytest.raises(RuntimeError, match='bootstrap'):
        r.retrieve('sales')


def test_no_keyword_evidence_does_not_manufacture_rrf_votes(make_retriever):
    r, _ = make_retriever()
    r.ensure_index()
    query = 'zzzzzzzzzzzz'
    keyword = r.retrieve_with_usage(query, strategy='keyword', use_cache=False)
    vector = r.retrieve_with_usage(query, strategy='vector', use_cache=False)
    hybrid = r.retrieve_with_usage(query, strategy='hybrid', use_cache=False)
    legacy = r.retrieve_with_usage(query, strategy='legacy_hybrid', use_cache=False)
    assert keyword.hits == []
    assert hybrid.hits == vector.hits
    assert sum(h.score for h in legacy.hits) > sum(h.score for h in hybrid.hits)

from types import SimpleNamespace as NS

import pytest

from app.config import Settings
from app.embeddings import EmbeddingProvider
from app.rag import HybridRetriever


def config(**kwargs):
    return Settings(_env_file=None, **({"embedding_provider": "openai", "embedding_api_key": "test",
                    "embedding_dimensions": 2} | kwargs))


def response(vectors, indices=None):
    indices = indices if indices is not None else list(range(len(vectors)))
    return NS(data=[NS(index=i, embedding=v) for i, v in zip(indices, vectors)], usage=NS(total_tokens=7))


def client(responses):
    values = iter(responses)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        result = next(values)
        if isinstance(result, Exception):
            raise result
        return result
    return NS(embeddings=NS(create=create), calls=calls)


def test_batch_limit_reorders_response_and_sums_usage():
    c = client([response([[0, 1], [1, 0]], [1, 0]), response([[1, 1]])])
    p = EmbeddingProvider(config(embedding_batch_size=2), client=c)
    result = p.embed(["one", "two", "three"])
    assert result.vectors == [[1, 0], [0, 1], [1, 1]]
    assert result.tokens == 14
    assert [len(call['input']) for call in c.calls] == [2, 1]
    assert all(call['dimensions'] == 2 and call['encoding_format'] == 'float' for call in c.calls)


@pytest.mark.parametrize('bad', [response([]), response([[1, 0]], [1]),
    response([[1, 0], [0, 1]], [0, 0]), response([[1]]), response([[0, 0]]),
    response([[float('nan'), 1]]), response([[float('inf'), 1]])])
def test_malformed_responses_are_rejected(bad):
    with pytest.raises(ValueError, match='Embedding response'):
        EmbeddingProvider(config(), client=client([bad])).embed(['query'])


def test_api_failure_is_not_silently_replaced_with_hash():
    with pytest.raises(RuntimeError, match='provider unavailable'):
        EmbeddingProvider(config(), client=client([RuntimeError('provider unavailable')])).embed(['query'])


def test_missing_key_fails_before_request():
    with pytest.raises(ValueError, match='EMBEDDING_API_KEY'):
        EmbeddingProvider(config(embedding_api_key=''))


def test_hash_mode_needs_no_api_key():
    result = EmbeddingProvider(config(embedding_provider='hash', embedding_api_key='', embedding_dimensions=256)).embed(['销售额'])
    assert len(result.vectors[0]) == 256 and result.tokens == 0


def test_empty_inputs_do_not_call_api():
    c = client([])
    p = EmbeddingProvider(config(), client=c)
    assert p.embed([]).vectors == []
    with pytest.raises(ValueError, match='blank'):
        p.embed([' '])
    assert not c.calls


def test_index_and_cache_version_changes_with_model_dimensions_endpoint_and_content(tmp_path):
    corpus = tmp_path / 'knowledge.yaml'
    corpus.write_text('- id: a\n  title: one\n  content: first\n')
    def make(**updates):
        s = config(**updates)
        return HybridRetriever(corpus, settings=s, provider=EmbeddingProvider(s, client=client([])), redis_client=NS())
    baseline = make()
    key = baseline.cache_key('query', 5, None, 'hybrid')
    for update in ({'embedding_model': 'new'}, {'embedding_dimensions': 3},
                   {'embedding_base_url': 'https://example.com/v1'}, {'embedding_provider': 'hash'}):
        changed = make(**update)
        assert changed.index_version != baseline.index_version
        assert changed.cache_key('query', 5, None, 'hybrid') != key
    # Rotating a credential must not require reindexing.
    assert make(embedding_api_key='rotated').index_version == baseline.index_version
    corpus.write_text('- id: a\n  title: one\n  content: second\n')
    assert make().index_version != baseline.index_version
    assert baseline.cache_key('query', 5, {'metric'}, 'hybrid') != key
    assert baseline.cache_key('query', 5, None, 'vector') != key


def test_legacy_and_canonical_chat_configuration(monkeypatch):
    for key in ('LLM_API_KEY', 'DASHSCOPE_API_KEY', 'LLM_BASE_URL', 'DASHSCOPE_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    legacy = Settings(_env_file=None, DASHSCOPE_API_KEY='legacy', DASHSCOPE_BASE_URL='https://old.example/v1')
    assert legacy.dashscope_api_key == 'legacy'
    canonical = Settings(_env_file=None, LLM_API_KEY='canonical', LLM_BASE_URL='https://new.example/v1')
    assert canonical.dashscope_api_key == 'canonical'
    assert canonical.dashscope_base_url == 'https://new.example/v1'

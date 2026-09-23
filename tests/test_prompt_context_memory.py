from app.context import MemoryExtractor
from app.prompts import PromptRegistry
from app.rag import HybridRetriever


def test_prompt_registry_is_versioned():
    registry = PromptRegistry()
    rendered = registry.render('agent', context='测试上下文', sql_policy='只读SQL', token_budget=20000, max_loops=5)
    assert rendered.version == '1.2.0'
    assert len(rendered.prompt_hash) == 64
    assert '测试上下文' in rendered.text


def test_memory_only_extracts_explicit_preferences():
    extractor = MemoryExtractor()
    assert extractor.extract('帮我看看销售额') == []
    result = extractor.extract('以后默认用折线图，金额保留2位')
    values = {item.key: item.value for item in result}
    assert values['chart_type'] == '折线图'
    assert values['decimal_places'] == '2'


def test_hybrid_rag_retrieves_metric():
    from app.config import get_settings
    settings = get_settings().model_copy(update={"embedding_provider": "hash", "embedding_dimensions": 256})
    retriever = HybridRetriever(settings=settings)
    retriever.ensure_index()
    ids = [item.id for item in retriever.retrieve('销售额的计算口径', top_k=5)]
    assert 'metric_sales' in ids

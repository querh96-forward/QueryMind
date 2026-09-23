from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frontend_contains_business_pages():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for label in ["智能分析", "经营看板", "系统表现", "数据源", "指标口径", "设置"]:
        assert label in html


def test_frontend_hides_internal_agent_debugging():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    for label in ["LangGraph节点", "Prompt快照", "Tool Call参数", "隐藏思维链"]:
        assert label not in html


def test_single_deployment_architecture():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service in ["postgres:", "redis:", "worker:", "api:", "web-mcp:", "querymind-mcp:"]:
        assert service in compose
    assert "sqlite" not in compose.lower()

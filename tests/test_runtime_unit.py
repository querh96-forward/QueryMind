"""Run the actual LangGraph/ToolRegistry without a database or paid model."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app.graph as graph_module
from app.config import Settings
from app.graph import AnalysisGraph
from app.model import FunctionCall, ModelDecision
from app.prompts import PromptRegistry
from app.tool_errors import SQLInputError, SQLPolicyError
from app.tools import ToolRegistry


def call(sql, identifier="sql"):
    return ModelDecision(tool_calls=[FunctionCall(identifier, "query_database", {"sql": sql})])


class ScriptedModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.inputs = []

    def invoke(self, system_prompt, messages, tools, on_token=None):
        pending = set()
        for item in messages:
            if item["role"] == "assistant":
                assert not pending, "previous tool calls were not answered"
                pending.update(call["id"] for call in item.get("tool_calls", []))
            elif item["role"] == "tool":
                pending.remove(item["tool_call_id"])
        assert not pending, "model received an incomplete tool-result batch"
        self.inputs.append(deepcopy(messages))
        assert self.decisions, "unexpected additional model call"
        return self.decisions.pop(0)


@pytest.fixture
def harness(monkeypatch):
    def build(decisions, query=None, **limits):
        settings = Settings(_env_file=None, dashscope_api_key="", mock_llm=True,
                            **({"max_agent_loops": 12, "max_tool_calls": 20} | limits))
        monkeypatch.setattr(graph_module, "get_settings", lambda: settings)
        monkeypatch.setattr(graph_module, "CheckpointManager", lambda: SimpleNamespace(saver=InMemorySaver()))
        updates, audit, events, executed = [], [], [], []
        record = SimpleNamespace(status="running")
        repo = SimpleNamespace(
            get_run=lambda _: record,
            update_run=lambda _, **fields: updates.append(fields),
            add_tool_call=lambda *args: audit.append(args),
            save_prompt_snapshot=lambda *args: None,
            upsert_memory=lambda *args, **kwargs: None,
        )

        def execute(sql):
            executed.append(sql)
            if query:
                return query(sql)
            return {"sql": sql, "rows": [{"销售额": 12}], "row_count": 1}

        source = SimpleNamespace(query=execute, schema=lambda **_: [])
        registry = ToolRegistry(source, mcp_search=SimpleNamespace(available=False))
        model = ScriptedModel(decisions)
        pack = SimpleNamespace(schema="test schema", estimated_tokens=10, prompt_text=lambda: "test context")
        graph = AnalysisGraph(repo, SimpleNamespace(build=lambda **_: pack), PromptRegistry(), model,
                              registry, SimpleNamespace(publish=lambda *args, **kwargs: events.append(args)))
        initial = {"run_id": "run", "thread_id": "thread", "user_id": "user",
                   "question": "查询销售额", "token_budget": 20_000}
        return SimpleNamespace(graph=graph, initial=initial, model=model, executed=executed,
                               audit=audit, updates=updates, events=events, record=record)
    return build


def test_first_sql_failure_survives_repair_and_schema_lookup(harness):
    def query(sql):
        if sql == "bad":
            raise SQLInputError("syntax error")
        return {"sql": sql, "rows": [{"销售额": 12}]}
    h = harness([call("bad"), ModelDecision(tool_calls=[FunctionCall("schema", "inspect_schema", {})]),
                 call("fixed", "sql2"), ModelDecision(content="销售额12")], query)
    result = h.graph.invoke(h.initial)
    assert result["verification_passed"] is True
    assert result["first_sql_success"] is False
    assert result["sql_failure_count"] == 1
    assert h.updates == [{"first_sql_success": False}]
    assert result["answer_payload"]["table"]["rows"] == [{"销售额": 12}]


def test_permission_denial_stops_and_answers_remaining_batch(harness):
    def query(sql):
        raise SQLPolicyError("write is forbidden")
    decision = ModelDecision(tool_calls=[FunctionCall("a", "query_database", {"sql": "DELETE"}),
                                          FunctionCall("b", "inspect_schema", {})])
    h = harness([decision], query)
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "permission_denied"
    assert not result["verification_passed"]
    replies = [item for item in result["messages"] if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in replies] == ["a", "b"]
    assert replies[-1]["content"]["skipped"] is True
    assert len(h.executed) == 1
    assert len(h.model.inputs) == 1


def test_tool_limit_completes_batch_without_executing_excess(harness):
    decision = ModelDecision(tool_calls=[FunctionCall(str(i), "query_database", {"sql": str(i)}) for i in range(3)])
    h = harness([decision], max_tool_calls=1)
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "tool_limit"
    assert h.executed == ["0"]
    assert len([item for item in result["messages"] if item["role"] == "tool"]) == 3
    assert len(h.audit) == 3
    assert result["tool_call_count"] == 1
    assert not result["verification_passed"]


def test_exact_tool_limit_still_allows_final_answer(harness):
    h = harness([call("select"), ModelDecision(content="结果是12")], max_tool_calls=1)
    assert h.graph.invoke(h.initial)["verification_passed"] is True


def test_repeated_call_stops_without_third_execution(harness):
    h = harness([call("same", str(i)) for i in range(3)])
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "repeated_call"
    assert h.executed == ["same", "same"]
    assert result["loop_count"] == 3


def test_sql_repair_limit_caps_distinct_failing_queries(harness):
    def fail(sql):
        raise SQLInputError("bad syntax")
    h = harness([call(str(i), str(i)) for i in range(3)], fail, max_sql_repairs=2)
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "sql_repair_limit"
    assert len(h.executed) == 3
    assert result["first_sql_success"] is False


def test_failed_sql_cannot_be_masked_by_successful_schema_tool(harness):
    def query(sql):
        if sql == "bad":
            raise SQLInputError("bad syntax")
        return {"sql": sql, "rows": []}
    h = harness([call("bad"), ModelDecision(tool_calls=[FunctionCall("s", "inspect_schema", {})]),
                 ModelDecision(content="声称已完成"), call("fixed", "q2"), ModelDecision(content="无匹配数据")], query)
    result = h.graph.invoke(h.initial)
    assert len(h.model.inputs) == 5
    assert "上一步未通过验证" in h.model.inputs[3][-1]["content"]
    assert result["verification_passed"] is True


def test_schema_lookup_after_success_preserves_table(harness):
    h = harness([call("select"), ModelDecision(tool_calls=[FunctionCall("s", "inspect_schema", {})]),
                 ModelDecision(content="销售额12")])
    result = h.graph.invoke(h.initial)
    assert result["answer_payload"]["table"]["rows"] == [{"销售额": 12}]


def test_empty_rows_are_valid_and_not_retried(harness):
    h = harness([call("empty"), ModelDecision(content="没有匹配记录")], lambda sql: {"sql": sql, "rows": []})
    result = h.graph.invoke(h.initial)
    assert result["verification_passed"] is True
    assert len(h.executed) == 1


def test_loop_cap_is_not_a_successful_answer(harness):
    h = harness([call("select")], max_agent_loops=1)
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "loop_limit"
    assert result["loop_count"] == 1
    assert not result["verification_passed"]


def test_token_exhaustion_blocks_tools_without_replan(harness):
    decision = call("select")
    decision.input_tokens = 20_000
    h = harness([decision])
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "token_budget"
    assert h.executed == []
    assert len(h.model.inputs) == 1


def test_cancel_before_first_call_avoids_model_and_tools(harness):
    h = harness([])
    h.record.status = "canceled"
    result = h.graph.invoke(h.initial)
    assert result["stop_reason"] == "canceled"
    assert h.executed == []


def test_stale_preface_is_not_accepted_as_final_answer(harness):
    decision = call("select")
    decision.content = "我即将查询"
    h = harness([decision, ModelDecision(content="")], max_agent_loops=2)
    result = h.graph.invoke(h.initial)
    assert not result["verification_passed"]
    assert result["answer_text"] != "我即将查询"


def test_runtime_limit_prevents_replanning(harness):
    h = harness([])
    state = h.initial | {"started_at_epoch": 0, "loop_count": 0}
    result = h.graph._model(state)
    assert result["stop_reason"] == "runtime_limit"
    assert not result["should_replan"]


def test_invalid_tool_arguments_never_reach_handler(harness):
    h = harness([])
    result = h.graph.tools.execute("run", "query_database", {"sql": 123}, h.graph.repo)
    assert result["error_type"] == "invalid_arguments"
    assert result["retryable"] is True
    assert h.executed == []

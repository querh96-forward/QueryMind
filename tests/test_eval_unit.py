from types import SimpleNamespace

import pytest

from evals.run_eval import evaluate_case
from evals.scoring import compare_rows, summarize


def test_unordered_comparison_preserves_duplicate_counts():
    actual = [{"x": 1}, {"x": 2}, {"x": 1}]
    expected = [{"x": 1}, {"x": 1}, {"x": 2}]
    assert compare_rows(actual, expected, ordered=False)["correct"]
    assert not compare_rows(actual, expected)["correct"]
    assert not compare_rows(actual, [{"x": 1}, {"x": 2}, {"x": 2}], ordered=False)["correct"]


def test_tolerance_matching_does_not_greedily_consume_wrong_row():
    assert compare_rows([{"x": .1}, {"x": 0}], [{"x": 0}, {"x": .2}],
                        ordered=False, abs_tolerance=".11")["correct"]


@pytest.mark.parametrize("a,b,correct", [(1, 1.0, True), (1.004, 1, True), (1.006, 1, False),
                                        (None, 0, False), (True, 1, False), ("1", 1, False)])
def test_numeric_and_null_semantics(a, b, correct):
    assert compare_rows([{"x": a}], [{"x": b}])["correct"] is correct


def test_aliases_are_explicit_and_cannot_merge_columns():
    assert not compare_rows([{"total": 1}], [{"销售额": 1}])["correct"]
    assert compare_rows([{"total": 1}], [{"销售额": 1}], column_aliases={"total": "销售额"})["correct"]
    assert not compare_rows([{"total": 1, "销售额": 1}], [{"销售额": 1}],
                            column_aliases={"total": "销售额"})["correct"]


@pytest.mark.parametrize("status", ["failed", "completed_with_warning", "canceled", "running"])
def test_empty_reference_cannot_make_unsuccessful_run_pass(status):
    record = SimpleNamespace(status=status, answer={}, first_sql_success=None, latency_ms=0,
                             input_tokens=0, output_tokens=0, embedding_tokens=0, tool_call_count=0)
    canceled = []
    service = SimpleNamespace(
        source=SimpleNamespace(query=lambda _: {"rows": []}),
        repo=SimpleNamespace(create_thread=lambda **_: SimpleNamespace(id="t"), get_run=lambda _: record),
        create_analysis=lambda **_: "r", cancel_run=lambda rid: canceled.append(rid),
    )
    result = evaluate_case(service, {"id": "x", "question": "empty", "reference_sql": "SELECT"}, 0)
    assert result["correct"] is False
    if status == "running":
        assert result["status"] == "eval_timeout"
        assert canceled == ["r"]


def test_successful_empty_query_is_valid():
    record = SimpleNamespace(status="completed", answer={"evidence": {"sql": "SELECT"}, "table": {"rows": []}},
                             first_sql_success=True, latency_ms=1, input_tokens=1, output_tokens=1,
                             embedding_tokens=0, tool_call_count=1)
    service = SimpleNamespace(source=SimpleNamespace(query=lambda _: {"rows": []}),
                              repo=SimpleNamespace(create_thread=lambda **_: SimpleNamespace(id="t"), get_run=lambda _: record),
                              create_analysis=lambda **_: "r")
    assert evaluate_case(service, {"id": "x", "question": "empty", "reference_sql": "SELECT"}, 1)["correct"]


def test_metric_denominators_and_unavailable_recovery():
    base = {"correct": True, "status": "completed", "latency_ms": 10, "total_tokens": 20, "tool_call_count": 1}
    metrics = summarize([base | {"first_sql_success": True}, base | {"first_sql_success": None}])
    assert metrics["first_sql_success_rate"] == 100
    assert metrics["sql_attempted_case_count"] == 1
    assert metrics["recovery_result_accuracy"] is None
    metrics = summarize([base | {"first_sql_success": False},
                         base | {"first_sql_success": False, "correct": False}])
    assert metrics["recovery_result_accuracy"] == 50

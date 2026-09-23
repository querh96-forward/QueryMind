from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.scoring import compare_rows, summarize

TERMINAL = {"completed", "completed_with_warning", "failed", "canceled"}


def evaluate_case(service, case: dict, timeout: float) -> dict:
    # A bad reference query is a dataset error, not a model failure.
    expected = service.source.query(case["reference_sql"])["rows"]
    thread = service.repo.create_thread(title="离线评测")
    started = time.monotonic()
    run_id = service.create_analysis(thread_id=thread.id, question=case["question"])
    while True:
        record = service.repo.get_run(run_id)
        if record.status in TERMINAL:
            status = record.status
            break
        if time.monotonic() - started >= timeout:
            status = "eval_timeout"
            service.cancel_run(run_id)
            break
        time.sleep(0.2)
    latency = round((time.monotonic() - started) * 1000, 2)
    answer = record.answer or {}
    execution = answer.get("execution", {})
    actual = answer.get("table", {}).get("rows", [])
    options = case.get("comparison", {})
    comparison = compare_rows(actual, expected, **options)
    # Failed/unfinished runs must never match an empty reference result.
    valid = status == "completed" and bool(answer.get("evidence", {}).get("sql"))
    if not valid:
        comparison = {"correct": False, "reason": "run_not_completed_with_sql_evidence"}
    return {
        "id": case["id"], "question": case["question"], "category": case.get("category", "legacy"),
        "split": case.get("split", "development"), "run_id": run_id, "status": status,
        **comparison, "first_sql_success": record.first_sql_success,
        "latency_ms": latency, "worker_latency_ms": record.latency_ms,
        "total_tokens": record.input_tokens + record.output_tokens + record.embedding_tokens,
        "embedding_tokens": record.embedding_tokens,
        "tool_call_count": record.tool_call_count, "stop_reason": execution.get("stop_reason", ""),
        "sql_failure_count": execution.get("sql_failure_count", 0),
        "actual_sql": answer.get("evidence", {}).get("sql", ""),
        "expected_rows": expected, "actual_rows": actual, "comparison": options,
    }


def revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_hash() -> str:
    digest = hashlib.sha256()
    for pattern in ("app/**/*.py", "prompts/*.yaml", "evals/*.py", "requirements.txt"):
        for path in sorted(ROOT.glob(pattern)):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(dataset: Path, output: Path, *, require_real: bool = False, timeout: float = 180,
        split: str = "all") -> dict[str, Any]:
    from app.config import get_settings
    from app.service import AnalysisService

    settings = get_settings()
    if require_real and not settings.real_llm_enabled:
        raise RuntimeError("本次要求真实模型，但未启用真实API。检查密钥与MOCK_LLM。")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    raw = dataset.read_bytes()
    cases = json.loads(raw)
    cases = [case for case in cases if split == "all" or case.get("split", "development") == split]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("评测集不能为空，case id必须唯一")
    service = AnalysisService()
    mode = "real_api" if settings.real_llm_enabled else "mock"
    name = f"{dataset.stem} / {split} / {mode}"
    report: dict[str, Any] = {
        "dataset": name, "mode": mode, "model": settings.llm_model if mode == "real_api" else "Mock",
        "created_at": datetime.now(timezone.utc).isoformat(), "case_count": len(cases),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "git_revision": revision(),
        "source_sha256": source_hash(), "complete": False,
        "retrieval": service.retriever.index_info(),
        "limits": {"loops": settings.max_agent_loops, "tools": settings.max_tool_calls,
                   "sql_repairs": settings.max_sql_repairs, "tokens": settings.default_token_budget,
                   "runtime_seconds": settings.max_runtime_seconds, "eval_timeout_seconds": timeout},
        "metric_scope": "SQL result-table correctness; narrative correctness is not graded",
        "latency_scope": "submission through terminal status, including queue and polling",
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(output)

    save()
    for case in cases:
        item = evaluate_case(service, case, timeout)
        report["cases"].append(item)
        report["metrics"] = summarize(report["cases"])
        save()
        print(f"{item['id']}: {item['status']} correct={item['correct']} ({item['reason']})", flush=True)

    rag_cases = json.loads((ROOT / "evals/rag_cases.json").read_text(encoding="utf-8"))
    ranks = []
    for case in rag_cases:
        ids = [hit.id for hit in service.retriever.retrieve(case["query"], top_k=5)]
        ranks.append(1 / (ids.index(case["expected_id"]) + 1) if case["expected_id"] in ids else 0)
    report["metrics"].update({
        "rag_recall_at_5": round(sum(rank > 0 for rank in ranks) / len(ranks), 4) if ranks else None,
        "rag_mrr": round(sum(ranks) / len(ranks), 4) if ranks else None,
    })
    report["metrics_by_split"] = {
        label: summarize([item for item in report["cases"] if item["split"] == label])
        for label in sorted({item["split"] for item in report["cases"]})
    }
    report["complete"] = True
    service.repo.save_eval_run(name, report["model"], report["metrics"], len(cases))
    save()
    print(json.dumps({"report": str(output), "mode": mode, "metrics": report["metrics"]}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible QueryMind SQL-result evaluation")
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/portfolio_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/latest_report.json")
    parser.add_argument("--require-real", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--split", choices=["all", "development", "heldout"], default="all")
    args = parser.parse_args()
    run(args.dataset, args.output, require_real=args.require_real, timeout=args.timeout, split=args.split)

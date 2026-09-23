"""Paired retrieval evaluation: unchanged corpus, RRF and labels; cache disabled."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import ROOT, get_settings
from app.rag import HybridRetriever
from evals.run_eval import source_hash


def summarize(rows):
    n = len(rows)
    return {
        "count": n,
        "recall_at_1": round(sum(r["rank"] == 1 for r in rows) / n, 4),
        "recall_at_5": round(sum(r["rank"] > 0 for r in rows) / n, 4),
        "mrr_at_5": round(sum(1 / r["rank"] if r["rank"] else 0 for r in rows) / n, 4),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 2),
        "embedding_tokens": sum(r["embedding_tokens"] for r in rows),
    }


def run(output: Path):
    settings = get_settings()
    if settings.embedding_provider != "openai":
        raise RuntimeError("This comparison requires real semantic embeddings")
    retrievers = {
        "hash256": HybridRetriever(settings=settings.model_copy(update={
            "embedding_provider": "hash", "embedding_dimensions": 256})),
        "semantic": HybridRetriever(settings=settings),
    }
    datasets = {}
    cases = []
    for label, filename in (("legacy", "rag_cases.json"), ("paraphrase", "semantic_cases.json"),
                            ("heldout", "semantic_heldout_cases.json")):
        raw = (ROOT / "evals" / filename).read_bytes()
        datasets[label] = {"path": filename, "sha256": hashlib.sha256(raw).hexdigest()}
        for i, case in enumerate(json.loads(raw)):
            cases.append({**case, "id": case.get("id", f"legacy_{i + 1:02}"), "group": label})
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "complete": False,
        "source_sha256": source_hash(), "datasets": datasets, "cache_enabled": False,
        "scope": "One labelled target per query; Recall@5 and MRR@5, no answer generation.",
        "label_policy": "12 legacy + 20 diagnostic paraphrases; 16 new heldout queries fixed before the final run. Labels unchanged.",
        "change": "Real embeddings plus exclusion of zero-similarity keyword candidates; legacy_hybrid retains original zero-score padding for control.",
        "indexes": {name: r.ensure_index() for name, r in retrievers.items()},
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(output)

    save()
    for case in cases:
        for name, retriever in retrievers.items():
            for strategy in ("legacy_hybrid", "hybrid", "vector"):
                started = time.monotonic()
                result = retriever.retrieve_with_usage(case["query"], top_k=5, strategy=strategy, use_cache=False)
                ids = [h.id for h in result.hits]
                rank = ids.index(case["expected_id"]) + 1 if case["expected_id"] in ids else 0
                report["cases"].append({**case, "variant": f"{name}_{strategy}", "ids": ids, "rank": rank,
                                        "embedding_tokens": result.embedding_tokens,
                                        "latency_ms": round((time.monotonic() - started) * 1000, 2)})
        save()
        print(f"{case['id']} evaluated", flush=True)
    report["metrics"] = {}
    for variant in sorted({r["variant"] for r in report["cases"]}):
        report["metrics"][variant] = {}
        for group in ("all", "legacy", "paraphrase", "heldout"):
            rows = [r for r in report["cases"] if r["variant"] == variant and (group == "all" or r["group"] == group)]
            report["metrics"][variant][group] = summarize(rows)
    report["complete"] = True
    save()
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evals/reports/semantic-retrieval.json")
    args = parser.parse_args()
    run(args.output)

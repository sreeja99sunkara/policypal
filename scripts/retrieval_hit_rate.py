"""Retrieval hit rate over eval/eval_set.json (local only, no LLM calls).

For each answerable item whose "source" names page number(s), retrieve the
top 5 chunks and count a hit when any comes from one of those pages.

"source" pages are 1-indexed viewer pages; chunk metadata is 0-indexed, so we
subtract 1 before comparing.

Usage:
    python scripts/retrieval_hit_rate.py <label> [--mode MODE]
MODE (defaults to the label when the label is a mode name, else "app"):
    vector_top5    vector similarity top 5, no reranker
    vector_rerank  vector top 20 -> cross-encoder -> top 5
    hybrid_rerank  hybrid (vector+BM25, RRF) top 25 -> cross-encoder -> top 5
    app            whatever rag_chain.retrieve_and_rerank currently does
Writes eval/hit_rate_<label>.json
"""
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Local only: load embedding / reranker models from the HF cache, never the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.services.hybrid import hybrid_search  # noqa: E402
from app.services.rag_chain import retrieve_and_rerank  # noqa: E402
from app.services.reranker import rerank  # noqa: E402
from app.services.vectorstore import get_vectorstore  # noqa: E402

EVAL_PATH = ROOT / "eval" / "eval_set.json"
TOP_K = 5

MODES = {
    "vector_top5": lambda q: get_vectorstore().similarity_search(q, k=TOP_K),
    "vector_rerank": lambda q: rerank(q, get_vectorstore().similarity_search(q, k=20), top_n=TOP_K),
    "hybrid_rerank": lambda q: rerank(q, hybrid_search(q, k=20)[:25], top_n=TOP_K),
    "app": retrieve_and_rerank,
    # current production path: hybrid -> rerank top 5 -> attach table headers (<= 6 chunks)
    "pipeline": retrieve_and_rerank,
}
# Modes scored on everything they return (the final LLM context), not just the first 5.
FULL_CONTEXT_MODES = {"app", "pipeline"}


def parse_pages(source: str | None) -> list[int]:
    """'Page 7, B(18)' -> [7]; 'Pages 6, 19, B(6)' -> [6, 19]; 'Pages 13-14, D(7)' -> [13, 14];
    'Pages 18/19' -> [18, 19]; 'Pages 5 and 19' -> [5, 19]."""
    if not source:
        return []
    m = re.search(r"Pages?\s+((?:\d+|\s|,|/|-|–|&|\band\b)+)", source)
    if not m:
        return []
    pages: list[int] = []
    for part in re.split(r",|/|&|\band\b", m.group(1)):
        part = part.strip()
        if not part:
            continue
        rng = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", part)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            pages.extend(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            pages.append(int(part))
    return sorted(set(pages))


def chunk_pages(doc) -> list[int]:
    """0-indexed pages of a chunk; merged duplicates carry e.g. pages="18,19"."""
    pages = doc.metadata.get("pages")
    if pages:
        return [int(p) for p in str(pages).split(",") if p.strip().isdigit()]
    return [doc.metadata.get("page")]


def main() -> None:
    args = sys.argv[1:]
    if not args:
        sys.exit("usage: python scripts/retrieval_hit_rate.py <label> [--mode MODE]")
    label = args[0]
    mode = args[args.index("--mode") + 1] if "--mode" in args else (label if label in MODES else "app")
    if mode not in MODES:
        sys.exit(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
    retrieve = MODES[mode]
    print(f"label={label} mode={mode}")

    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["items"]
    results, skipped, latencies = [], [], []
    retrieve("warm-up query")  # load models before timing
    for item in items:
        if not item.get("answerable"):
            continue
        pages_1 = parse_pages(item.get("source"))
        if not pages_1:
            skipped.append(item["id"])
            continue
        expected = {p - 1 for p in pages_1}  # 1-indexed -> 0-indexed
        t0 = time.perf_counter()
        docs = retrieve(item["question"])
        latencies.append(time.perf_counter() - t0)
        if mode not in FULL_CONTEXT_MODES:
            docs = docs[:TOP_K]
        got = [chunk_pages(d) for d in docs]
        hit = any(p in expected for pages in got for p in pages)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "source": item["source"],
            "expected_pages_0idx": sorted(expected),
            "retrieved_pages_0idx": [p[0] if len(p) == 1 else p for p in got],
            "n_chunks": len(docs),
            "retrieved_chunk_types": [d.metadata.get("chunk_type", "legacy") for d in docs],
            "hit": hit,
        })
        shown = [p[0] if len(p) == 1 else "/".join(map(str, p)) for p in got]
        print(f"[{'HIT ' if hit else 'MISS'}] #{item['id']:>2} expected={sorted(expected)} got={shown}  {item['question']}")

    n_hit = sum(r["hit"] for r in results)
    summary = {
        "label": label,
        "mode": mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mean_latency_ms": round(1000 * sum(latencies) / len(latencies), 1) if latencies else None,
        "top_k": TOP_K,
        "evaluated": len(results),
        "hits": n_hit,
        "hit_rate": round(n_hit / len(results), 4) if results else None,
        "skipped_no_page_in_source": skipped,
        "results": results,
    }
    out = ROOT / "eval" / f"hit_rate_{label}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nhit rate: {n_hit}/{len(results)} = {summary['hit_rate']}  "
          f"mean retrieval latency: {summary['mean_latency_ms']} ms")
    print(f"skipped (no page in source): {skipped}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

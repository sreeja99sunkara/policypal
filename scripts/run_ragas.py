"""Run RAGAS evaluation against the PolicyPal RAG chain (resumable).

Usage (from the policypal/ root, venv active, a policy PDF already indexed):
    python scripts/run_ragas.py --limit 3          # quick trial
    python scripts/run_ragas.py                    # full run
    python scripts/run_ragas.py --reuse-answers    # skip generation, only (re)score

Scores are cached in eval/score_cache.json keyed by judge model + question +
answer/context hash, so a rerun only fills the cells that are still empty
(for example after a Groq rate limit). Answers are saved to
eval/answers_latest.json so every response can be inspected.

Set GROQ_JUDGE_MODEL in .env to judge with a different model than the one
that generates the answers (separate Groq rate-limit bucket, less self-grading).
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the app uses relative paths like data/processed/chroma

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_groq import ChatGroq  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

from app.services.embedder import get_embeddings  # noqa: E402
from app.services.rag_chain import get_rag_chain  # noqa: E402
from app.services.vectorstore import get_vectorstore, has_documents  # noqa: E402

EVAL_DIR = ROOT / "eval"
EVAL_PATH = EVAL_DIR / "eval_set.json"
ANSWERS_PATH = EVAL_DIR / "answers_latest.json"
CACHE_PATH = EVAL_DIR / "score_cache.json"
K = 3  # must match the retriever k in rag_chain.py
METRICS = [
    "faithfulness",
    "answer_relevancy",
    "llm_context_precision_with_reference",
    "context_recall",
]
REFUSAL_PHRASES = ("don't have enough information", "do not have enough information")


def load_items(limit):
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["items"]
    ready = [i for i in items if not i["ground_truth"].strip().upper().startswith("TODO")]
    skipped = len(items) - len(ready)
    if skipped:
        print(f"Skipping {skipped} entries whose ground_truth is still TODO.")
    return ready[:limit] if limit else ready


def is_refusal(answer):
    a = answer.lower().replace("’", "'")
    return any(p in a for p in REFUSAL_PHRASES)


def valid(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def mean(values):
    vals = [v for v in values if valid(v)]
    return round(sum(vals) / len(vals), 3) if vals else None


def generate_answers(items):
    chain = get_rag_chain()
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": K})
    rows = []
    for n, item in enumerate(items, 1):
        q = item["question"]
        t0 = time.time()
        answer = chain.invoke(q)["answer"]
        latency = round(time.time() - t0, 2)
        # Same labelled text the model actually saw.
        contexts = [f"[Page {d.metadata['page']}]\n{d.page_content}" for d in retriever.invoke(q)]
        rows.append({**item, "response": answer, "contexts": contexts, "latency_s": latency})
        print(f"[{n}/{len(items)}] {latency:>5}s  {q}")
    return rows


def row_key(judge, r):
    h = hashlib.sha1(
        (r["response"] + "||" + "||".join(r["contexts"]) + "||" + r["ground_truth"]).encode("utf-8")
    ).hexdigest()[:12]
    return f"{judge}|{r['id']}|{h}"


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--skip-relevancy", action="store_true",
                        help="do not compute answer_relevancy (about 3x the tokens of the others)")
    parser.add_argument("--reuse-answers", action="store_true",
                        help="skip generation and reuse eval/answers_latest.json")
    args = parser.parse_args()
    active = [m for m in METRICS if not (args.skip_relevancy and m == "answer_relevancy")]
    metric_objs = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": ResponseRelevancy(),
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set. Check your .env file.")

    # ---- 1. Answers (generated, or reused from the last run) ----
    if args.reuse_answers and ANSWERS_PATH.exists():
        rows = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
        # Take ground truths, answerable flags and sources from the CURRENT eval set,
        # so edits to eval_set.json apply without regenerating answers.
        current = {i["id"]: i for i in json.loads(EVAL_PATH.read_text(encoding="utf-8"))["items"]}
        rows = [
            {**r, **{k: current[r["id"]][k] for k in ("question", "ground_truth", "answerable", "source")
                     if k in current[r["id"]]}}
            for r in rows if r["id"] in current
        ]
        if args.limit:
            rows = rows[: args.limit]
        print(f"Reusing {len(rows)} saved answers from {ANSWERS_PATH.name}")
    else:
        if not has_documents():
            sys.exit("No document is indexed. Upload the policy PDF first (via the API or the UI).")
        items = load_items(args.limit)
        if not items:
            sys.exit("No eval items with a filled-in ground_truth.")
        rows = generate_answers(items)
        ANSWERS_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    # ---- 2. Score only the cells that are still missing ----
    judge_model = os.getenv("GROQ_JUDGE_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    generator_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    print(f"\nGenerator: {generator_model} | Judge: {judge_model}")

    cache = load_cache()
    todo = [r for r in answerable
            if not all(valid(cache.get(row_key(judge_model, r), {}).get(m)) for m in active)]
    print(f"{len(answerable) - len(todo)} of {len(answerable)} questions already fully scored (cache).")

    if todo:
        # bypass_n: Groq rejects several completions per request, which answer
        # relevancy asks for by default.
        judge = LangchainLLMWrapper(
            ChatGroq(model=judge_model, temperature=0, api_key=os.getenv("GROQ_API_KEY")),
            bypass_n=True,
        )
        emb = LangchainEmbeddingsWrapper(get_embeddings())
        dataset = EvaluationDataset.from_list(
            [{"user_input": r["question"], "response": r["response"],
              "retrieved_contexts": r["contexts"], "reference": r["ground_truth"]} for r in todo]
        )
        result = evaluate(
            dataset,
            metrics=[metric_objs[m] for m in active],
            llm=judge,
            embeddings=emb,
            # Few retries: a daily token limit will not clear within a retry, so fail fast.
            run_config=RunConfig(max_workers=2, timeout=240, max_retries=3, max_wait=60),
            raise_exceptions=False,
        )
        df_new = result.to_pandas()
        for i, r in enumerate(todo):
            entry = cache.setdefault(row_key(judge_model, r), {})
            for m in active:
                v = df_new[m].iloc[i] if m in df_new else None
                if valid(v):
                    entry[m] = float(v)
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    # ---- 3. Merge, save, summarise ----
    merged = []
    for r in answerable:
        e = cache.get(row_key(judge_model, r), {})
        merged.append({"id": r["id"], "question": r["question"],
                       **{m: e.get(m) for m in METRICS}, "latency_s": r["latency_s"]})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    import pandas as pd  # noqa: E402
    pd.DataFrame(merged).to_csv(EVAL_DIR / f"results_{stamp}.csv", index=False)

    missing = {m: sum(not valid(x[m]) for x in merged) for m in active}
    summary = {
        "date": stamp,
        "generator_model": generator_model,
        "judge_model": judge_model,
        "questions_scored": len(answerable),
        "metrics_measured": active,
        "scores": {m: mean([x[m] for x in merged]) for m in active},
        "failed_metric_cells": missing,
        "complete": not any(missing.values()),
        "avg_latency_s": mean([r["latency_s"] for r in rows]),
        "unanswerable_total": len(unanswerable),
        "unanswerable_correctly_refused": sum(is_refusal(r["response"]) for r in unanswerable),
        "unanswerable_detail": {r["question"]: is_refusal(r["response"]) for r in unanswerable},
    }
    (EVAL_DIR / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== RAGAS summary =====")
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        print("\nINCOMPLETE: some cells are still empty (rate limit or unparseable judge output).\n"
              "Do not quote these numbers yet. Run again later with --reuse-answers; "
              "only the empty cells are redone.")
    else:
        print("\nComplete: every metric cell is filled. Safe to save as your baseline.")


if __name__ == "__main__":
    main()
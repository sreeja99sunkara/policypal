# eval/archive

Frozen copies of RAGAS evaluation artifacts. Nothing in this folder is read or
written by the scripts; do not overwrite these files.

Both runs used generator `openai/gpt-oss-120b` and judge `qwen/qwen3.8-27b`
(via Groq), 24 questions from `eval/eval_set.json`.

| File | What it is |
|---|---|
| `baseline_summary.json` | **Baseline** RAGAS summary, run `20261117_185415` (before table-aware parsing / hybrid / header attachment). faithfulness 0.953, llm_context_precision_with_reference 0.587, context_recall 0.555 (1 failed precision cell). |
| `baseline_results_20261117_185415.csv` | Per-question scores for the baseline run. |
| `answers_after_pipeline.json` | Generated answers + retrieved contexts from the after-pipeline run (copy of `eval/answers_latest.json` as of 2026-11-17 23:48). |
| `after_results_20261117_235950_INCOMPLETE.csv` | Per-question scores for the after-pipeline run `20261117_235950`. **INCOMPLETE** |
| `score_cache_backup.json` | Snapshot of `eval/score_cache.json` (last written 2026-11-17 23:59, after the after-pipeline run). |

**The after-pipeline run is incomplete and must not be quoted.** Most metric
cells failed because of Groq output-token limits (failed cells: faithfulness 18,
context precision 20, context recall 11 of 24), so its averages
(0.583 / 0.833 / 0.628) are computed over a small, non-comparable subset.

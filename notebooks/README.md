# Memory engine benchmark notebooks

Shared implementation (edit these, not notebook logic):

| File | Role |
|---|---|
| `memory_retrieval.py` | Staged hybrid retrieval + legacy-named Atlas/Episode Recall Route→Ground drill-down |
| `benchmark_harness.py` | Ledger, formation, Episode Recall (legacy `Atlas` names), compile, answer/judge, run loop |
| `dataset_locomo.py` | LoCoMo download, ingest, evidence@k |
| `dataset_longmemeval.py` | LongMemEval download, ingest, session-scoped retrieval |
| `dataset_halumem.py` | HaluMem download, ingest, QA + extraction recall |
| `dataset_memorydoc.py` | MemoryDocDataSet ingest (chat + long docs), hybrid evidence |

**Primary notebook:** `memory_benchmarks.ipynb` — LoCoMo, LongMemEval, HaluMem, MemoryDocDataSet.

Legacy: `ledger_lens_locomo.ipynb` redirects here.

## Data locations

| Benchmark | Default path | Download |
|---|---|---|
| LoCoMo | `data/locomo/locomo10.json` | Auto (GitHub raw) |
| LongMemEval | `data/longmemeval/` | Auto (Hugging Face cleaned split) |
| HaluMem | `data/halumem/HaluMem-Medium.jsonl` | Auto (Hugging Face) |
| MemoryDocDataSet | `data/memorydoc/` | Official JSON when released; `fixtures/smoke_micro_world.json` for smoke tests |

Run from repo root or `notebooks/` with project venv + `GEMINI_API_KEY`.

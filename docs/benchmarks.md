# Benchmark results

The active research workflow uses the pinned [OmniMemEval user-memory harness](https://github.com/MemTensor/OmniMemEval/tree/ff13e4a5b12e3c406a81c32bb0834c7e8d73f38e). Fresh results and their manifests go under `data/omnimemeval/runs/`. The submitted evaluation and prior local runs are retained in `data/archive/2026-11-20/`.

| Suite | Split | Primary shared-harness score |
| --- | --- | --- |
| LoCoMo | Categories 1–4, 1,540 questions | Judged QA accuracy |
| PersonaMem-v2 | Text / 32K, 5,000 questions | Exact multiple-choice accuracy |
| HaluMem | Medium | Judged QA accuracy |
| BEAM | Official 100K split, 400 questions | Nugget score |
| LongMemEval | S | Judged QA accuracy |

The optional BEAM-10M suite also uses the nugget rubric. Run commands, output locations, and resume options are in [Operations](operations.md#benchmarks).

See the [AAMAS rebuttal run plan](rebuttal-evaluation.md) for the ablation matrix, shared models, call counts, and budget estimates.

# AAMAS rebuttal evaluation run plan

Use LoCoMo for the component ablations. Start with a shared Trisynapse ingest, the full system, and raw BM25+dense with equal-weight reciprocal rank fusion (RRF). Run MemOS on the same LoCoMo harness when its cloud credentials are available. Then prioritize the PersonaMem-v2 pilot and BEAM-100K. Freeze the results by 23 November IST. The conference deadline is 25 November at 5:29 p.m. IST. [AAMAS dates](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/)

## Harness and models

All draft comparisons use OmniMemEval revision `ff13e4a5b12e3c406a81c32bb0834c7e8d73f38e`. The harness handles dataset iteration, context rendering, answering, and scoring. The Trisynapse adapter handles memory ingestion and retrieval. `top_k=20` is the default. LoCoMo makes two searches per question, one for each speaker, and hands both sections to the upstream reader. PersonaMem-v2, HaluMem, BEAM, and LongMemEval make one search per question. The plan starts with four retrieval workers and ten LLM workers per run. [OmniMemEval setup](https://github.com/MemTensor/OmniMemEval/blob/ff13e4a5b12e3c406a81c32bb0834c7e8d73f38e/docs/user_memory/results.md)

| Stage | Model or rule |
| --- | --- |
| Trisynapse fact extraction and group summaries | `gpt-4.1-mini-2025-04-14` |
| Trisynapse semantic retrieval | Local `sentence-transformers/all-MiniLM-L6-v2`. The BM25-only arm ranks lexical evidence. |
| Answer generation | `gpt-4.1-mini-2025-04-14` for every planned backend and dataset |
| Primary judge | `gpt-4o-mini-2024-07-18` for LoCoMo, HaluMem, BEAM, and LongMemEval |
| PersonaMem scoring | Upstream exact multiple-choice scoring |
| Optional second LoCoMo judge | `gpt-4.1-2025-04-14` on the saved full and raw-hybrid answers |
| MemOS cloud memory | Provider-managed. The external answer and judge models remain the same as for Trisynapse. |

The local environment contains an OpenAI credential. The MemOS cloud key still needs to be supplied. Provider authentication, account-specific rate limits, and actual billed token usage should be checked before launching the paid runs.

## Run order

1. Run `locomo-ingest` through stage 1. Use that snapshot for the 11 LoCoMo ablation arms. The runner copies the snapshot into each arm and records the source run, so retrieval differences can be attributed to the settings and reader or judge variation. Start with `locomo-full` and `locomo-raw-hybrid`.
2. Run `locomo-memos` when cloud access is available. Compare the same 1,540 LoCoMo questions, common reader, common judge, and per-speaker top-k. Start the other LoCoMo arms after the first two have valid search, answer, and judge coverage.
3. Run `personamem-pilot` and `personamem-memos-pilot` on the same first 16 personas in upstream order. This slice contains 478 questions, including 88 Ask-to-Forget and 66 Sensitive Information cases. Treat it as a pilot. If throughput and budget permit, run `personamem-full` and `personamem-memos-full` across all 200 personas. The full run is a new run and includes the pilot personas again.
4. Run `beam100k-full` as the long-history comparison at the budgeted scale. It has 20 conversations and 400 questions. Use the official 100K split and upstream nugget/event-ordering scoring. A cross-backend comparison requires a baseline on this same scale.
5. Run `halumem-qa` if time remains. This is OmniMemEval QA on HaluMem-Medium. The upstream runner loads all sessions for a user before QA, so the result describes retrieval over the supplied history. Native extraction, update scoring, and evaluation at each historical cutoff require their own protocol.
6. Defer LongMemEval-S and BEAM-10M until after the rebuttal. LongMemEval-S is staged. BEAM-10M has a preset and requires downloading and staging its separate split.

## LoCoMo ablations

All arms use the same seed, full LoCoMo split, and primary models. The raw baselines calculate lexical statistics or dense scores on raw text and export ranked raw records. Main component ablations each start from the full config.

| Arm | Treatment |
| --- | --- |
| `full` | Production retrieval and grounded readout |
| `raw-bm25` | Raw-text BM25 alone |
| `raw-dense` | Raw-text MiniLM cosine similarity alone |
| `raw-hybrid` | Raw-text BM25 and MiniLM cosine with equal-weight RRF |
| `no-graph` | Remove the graph route, similarity-edge preparation, and graph expansion during readout |
| `no-claims` | Exclude compiled claims while retaining extracted fact records |
| `no-summaries` | Exclude group-summary views and their deep-summary fallback |
| `no-reranker` | Bypass the candidate reranker. The grounded readout still scores and packs the evidence. |
| `no-rrf` | Fuse the same route rankings through deduplicated round-robin interleaving |
| `raw-readout` | Allow all projections to nominate evidence and export raw source records only |
| `ranked-readout` | Pack ranked candidates directly without Trace drilling or its scoring. Some candidates can be synthetic. Treat this as a research-only variant. |

The full versus ranked-readout comparison measures the combined Trace readout effect. A separate inspection of sampled fact records against their cited text is needed to measure entailment. For quality and complexity, record judged QA, per-category accuracy, unique gold evidence hit/recall when locators can be joined, context tokens, p50/p95 retrieval latency, and ingest cost. Report paired accuracy differences with intervals clustered by conversation. If the same corpus is reused to pick settings, describe the sensitivity results as diagnostic.

### Robustness and sensitivity

- Run `locomo-full-repeat-2`, `locomo-full-repeat-3`, `locomo-raw-hybrid-repeat-2`, and `locomo-raw-hybrid-repeat-3`. They use saved search contexts and run fresh answering and primary judging. They measure reader and judge variation at fixed ingestion and retrieval. The upstream `--num-runs` option repeats the judge on LoCoMo and the reader on PersonaMem-v2.
- Run `locomo-full-second-judge` and `locomo-raw-hybrid-second-judge` on the original answer files. Compare verdicts on the same question IDs and report raw agreement, Cohen’s κ, and both accuracies.
- Change one setting family at a time on the full config. Top-k is 16 or 24, the confidence floor and margin are scaled together by 0.8 or 1.2, and the RRF rank constant is 48 or 72. For a lower-cost retrieval sensitivity pass, append `--to-step 2` to any of these six presets. Report its retrieval and latency effects. QA sensitivity uses the whole pipeline.

## Expected calls and API spend

Counts come from the staged datasets and pinned harness’s message batching. Ingestion uses one extraction call and one summary call per nonempty group. Counts assume one answer and one primary judge repeat with successful requests. BEAM uses 878 judge calls for 400 questions because it scores by rubric item, with one event-ordering call per applicable question. Local search and MiniLM calls have no provider fee. Dollar figures are USD estimates for successful requests, before retries, cloud compute, taxes, and MemOS charges.

| Dataset or task | Extract | Summarize | Reader | Judge | Paid calls | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LoCoMo shared ingest | 1,696 | 1,696 | 0 | 0 | 3,392 | $1.3–3.4 |
| LoCoMo one arm using the snapshot | 0 | 0 | 1,540 | 1,540 | 3,080 | $1.0–4.3 |
| LoCoMo core, one ingest plus 11 arms | 1,696 | 1,696 | 16,940 | 16,940 | 37,272 | $12–50 |
| LoCoMo four reader and judge repeats | 0 | 0 | 6,160 | 6,160 | 12,320 | $4–17 |
| LoCoMo two second-judge runs | 0 | 0 | 0 | 3,080 | 3,080 | $3.1–9.5 |
| LoCoMo six sensitivity runs, scored | 0 | 0 | 9,240 | 9,240 | 18,480 | $6–26 |
| PersonaMem-v2 pilot, 478 questions | 561 | 561 | 478 | 0 | 1,600 | $1.2–2.5 |
| PersonaMem-v2 full, 5,000 questions | 7,040 | 7,040 | 5,000 | 0 | 19,080 | $14–29 |
| BEAM-100K | 748 | 748 | 400 | 878 | 2,774 | $3.4–5.4 |
| HaluMem-Medium QA | 8,058 | 8,058 | 3,467 | 3,467 | 23,050 | $11–29 |
| LongMemEval-S | 41,812 | 41,812 | 500 | 500 | 84,624 | $75–128 |

For MemOS, LoCoMo adds 1,540 reader and 1,540 judge calls, estimated at $1.0–4.3 for OpenAI. The pilot and full PersonaMem baseline add 478 and 5,000 reader calls respectively, estimated at $0.36–0.94 or $3.8–9.8. The harness makes approximately 544 logical MemOS add calls and 3,080 searches on LoCoMo. The PersonaMem pilot makes about 195 adds and 478 searches, and the full split makes 2,426 adds and 5,000 searches. Cloud batching and charges depend on the MemOS configuration and plan. BEAM-10M is outside the budgeted matrix. Its ingestion call count and rubric count should be measured after staging.

The initial LoCoMo seed and two primary Trisynapse arms are **$3.2–11.9** in API spend. The 11 core LoCoMo arms plus PersonaMem pilot and BEAM-100K are about **$17–58** before retry headroom. Budget **$20–70** for that package. The full LoCoMo program including repeats, second judging, and scored sensitivities is about **$25–103**, before the external baseline or the other suites.

### Pricing and throughput assumptions

- Standard API price per million input/output tokens is $0.40/$1.60 for GPT-4.1 mini, $0.15/$0.60 for GPT-4o mini, and $2.00/$8.00 for GPT-4.1. [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) · [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini) · [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1)
- Ingestion assumes four source characters per token, two passes over the source text, 500–1,300 additional input tokens per group across both calls, and 250–800 combined output tokens per group. The reader range is 1,200–5,600 input and 40–180 output tokens for LoCoMo, 1,800–4,500 input and 20–100 output for PersonaMem, 2,200–6,500 input and 70–400 output for BEAM, 1,800–5,500 input and 40–350 output for HaluMem, and 2,000–8,000 input and 80–350 output for LongMemEval. Judge calls use 350–900 input and 40–160 output. These are usage assumptions. Use `memory_model_calls.jsonl`, `token_usage_answer.json`, and `token_usage_eval.json` to reforecast from billed usage.
- At a sustained 150 model calls per minute, the model-call time alone is roughly 23 minutes for the shared LoCoMo ingest and 21 minutes per answer-and-judge arm. Local retrieval, writes, and stage scheduling add wall time. Allow roughly 3–8 hours for the seed and first two arms, then 8–24 hours for the remaining core arms if two arms can run concurrently. Allow 2–6 hours for the PersonaMem pilot and 2–8 hours for BEAM-100K. Previous PersonaMem retrieval took about 31 serial hours for 5,000 questions, and previous end-to-end HaluMem took about 38 hours. Decide the full runs from measured throughput in the new harness. Preserve at least 10 GiB of free space for LoCoMo arm copies and traces.

## Commands and artifacts

Use the project virtual environment from the repository root. `plan` validates the presets and prints commands without executing them. Select runs by name because the full catalog includes optional and deferred work.

```bash
trisynapse-memory bench omni plan
trisynapse-memory bench omni plan --name locomo-ingest
trisynapse-memory bench omni plan --name locomo-full
trisynapse-memory bench omni plan --name locomo-raw-hybrid
```

Run the printed commands in dependency order. `--reuse locomo-ingest --from-step 2` starts an arm from the saved ingestion. `--reuse locomo-full --from-step 3` repeats the reader and judge from saved searches. `--reuse locomo-full --from-step 4 --judge-model gpt-4.1-2025-04-14` rejudges identical answers. Use the corresponding retrieval config for reused search and answer stages. To resume the same named run, keep its settings and add `--resume`. Run outputs live in `data/omnimemeval/runs/<upstream-suite>/<backend>-<run-name>/`. The manifest records the run lineage, config, source hash, input hashes, and completed stage. [Run presets](../configs/omnimemeval/rebuttal.json) · [Operations](operations.md#benchmarks)

### Parallel execution and recovery

Preview a queue before starting paid requests. The `first` group runs shared LoCoMo ingestion and the two primary arms. The `fast` group also includes the remaining LoCoMo ablations, the PersonaMem pilot, and BEAM-100K. Add `--group robustness --group sensitivity` to include those groups under the same concurrency limit. Use repeated `--name` flags to select individual arms.

```bash
trisynapse-memory bench omni queue --group first --jobs 2 --llm-workers 6 --dry-run
trisynapse-memory bench omni queue --group first --jobs 2 --llm-workers 6
```

The queue includes prerequisites automatically and allows independent work to continue if another branch fails. Rerun the same command to resume unfinished work and skip completed stages. Each run records stage attempts in its manifest. LoCoMo also saves each completed judge decision within a conversation, so a failure on a later question preserves the earlier grades. Queue status and per-run logs are under `data/omnimemeval/queues/<group>/`. `--jobs` caps simultaneous experiments, and `--llm-workers` caps harness LLM workers per experiment. Start at two experiments and six LLM workers each. Adjust against observed throughput, provider limits, and CPU load.

Use direct requests for the rebuttal deadline. OpenAI Batch offers a 50% discount and preserves completed request results if a batch expires. Completed items can be reconciled by `custom_id` and only missing items need resubmission. Each dependent batch has its own turnaround window of up to 24 hours, so submitting ingestion, answering, and judging serially can delay downstream work. Batch orchestration remains a later option for work with more time, such as rejudging saved answers. [OpenAI Batch API guide](https://developers.openai.com/api/docs/guides/batch)

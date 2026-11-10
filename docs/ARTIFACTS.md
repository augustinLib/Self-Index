# Run outputs

## BRIGHT and table retrieval

`runs/` holds the log `run_<tag>.log` and the run directory `<domain>_pipe_<tag>/`:

| File | Contents |
|---|---|
| `run_config.json` | settings, parameters and models of the run |
| `queries.jsonl`, `queries_r<k>.jsonl` | the simulated queries of each iteration (`q`, the query, and `s`, its source document) |
| `round_log.json` | per-iteration counts and coverage |
| `keys.jsonl` | final generated keys: `{"doc": <id>, "keys": [...]}` |
| `per_query.json` | per-query nDCG@10 on the original index (base) and after the last iteration (evolved) |
| `summary.json` | base and evolved nDCG@10 |

## BrowseComp-Plus

Under `adapters/browsecomp/`:

| File | Contents |
|---|---|
| `agent/<tag>/keys.jsonl` | a published key set: one key (`k`) and its document (`docid`) per line |
| `agent/<tag>/key_emb.npz` | its Qwen3-Embedding-8B key embeddings |
| `indexes/bm25_<tag>/` | its BM25 index of the documents, each key and, per document, its keys joined into one entry |
| `indexes/bm25-concat_<tag>/` | its BM25 index of the documents and, per document, its keys joined into one entry |
| `runs/bcplus_pipe_<tag>/` | the index evolution |
| `runs/<tag>/<agent>-<dense, bm25 or bm25-concat>/` | one agent trajectory per query (`runs/base/` for the base index) |
| `evals/<tag>/<agent>-<dense, bm25 or bm25-concat>/evaluation_summary.json` | accuracy, evidence recall, search calls and calibration error |
| `evals/<tag>/<agent>-<dense, bm25 or bm25-concat>/final_report.json` | those metrics with token usage and cost |
| `runs/full_<tag>/` | the workspace of a `--stage full` run |

## LongMemEval-V2

Under `adapters/longmem/runs/`: the index evolution `lme_v2_small_pipe_<tag>/`, the
published keys `keys_<tag>.jsonl` and one directory per evaluation,
`<tag>/<memory system>/<web or enterprise>/`, whose `aggregated_metrics.json` holds the
reported number (`overall_full_set`).

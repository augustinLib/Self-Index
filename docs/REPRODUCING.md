# Reproducing the reported numbers

Every benchmark runs with the default parameters.

## Optimizer

One Qwen3.6-35B-A3B replica per GPU, each on its own port:

```
vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3.6-35b-a3b --port 8000 \
    --max-model-len 262144
export LLM_BASE_URL=http://localhost:8000/v1,http://localhost:8001/v1,...
```

## BRIGHT and table retrieval

```
python run.py --benchmark bright/<domain> --retriever <bge|bm25|bm25-concat|qwen3>
python run.py --benchmark table/<spider2|fiben|beaver> --retriever <bge|bm25|bm25-concat|qwen3>
```

The score is the `[EVAL] <domain> test nDCG@10 base=... evolved=...` line of
`runs/run_<tag>.log` and `summary.json` in `runs/<domain>_pipe_<tag>/`.

## BrowseComp-Plus

All 830 queries, judged by Qwen3-32B. The agents are GPT-OSS-120B
(`--agent-model gptoss`), GPT-5.4-nano (`nano54`), Gemini-3.7-Flash (`gem37`) and
Kimi-K2.5 (`kimi25`); the retrievers are Qwen3-Embedding-8B (`--retriever qwen3`) and
BM25 (`--retriever bm25` or `bm25-concat`). The indexes, queries and ground truth are
read from `adapters/browsecomp/` (`indexes/`, `topics-qrels/`, `data/`).

```
vllm serve Qwen/Qwen3-32B --max-model-len 40960           # judge
vllm serve openai/gpt-oss-120b --max-model-len 131072     # GPT-OSS-120B, one per GPU
export JUDGE_BASE_URL=<Qwen3-32B endpoint> AGENT_BASE_URL=<gpt-oss endpoints>
python run.py --benchmark bcplus --stage evolve --retriever <qwen3|bm25|bm25-concat> --tag <tag>
python run.py --benchmark bcplus --stage evaluate --retriever <qwen3|bm25|bm25-concat> --tag <tag> \
    --agent-model gptoss
```

The hosted agents need `AGENT_BASE_URL=https://api.openai.com/v1` with
`OPENAI_API_KEY` (GPT-5.4-nano), `GEMINI_API_KEY` (Gemini-3.7-Flash), and
`AGENT_BASE_URL=https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY` (Kimi-K2.5).
The base-index rows run an agent without keys (`dense` is Qwen3-Embedding-8B):

```
python run.py exec bc.agent --backbone <gptoss|nano54|gem37|kimi25> --retriever <dense|bm25>
```

## LongMemEval-V2 (small)

The index evolves over the memory entries of the Small haystacks with the
Qwen3-Embedding-8B memory encoder. The evaluation runs the official harness with
Qwen3.5-9B readers and the gpt-5.2 evaluator for each memory system of Table 4 on web
and enterprise. The data is read from `externals/LongMemEval-V2/data/longmemeval-v2/`.

```
vllm serve Qwen/Qwen3-Embedding-8B --runner pooling
vllm serve Qwen/Qwen3.5-9B --max-model-len 262144 --reasoning-parser qwen3
LLM_BASE_URL=<optimizer endpoints> EMBEDDING_BASE_URL=<Qwen3-Embedding-8B endpoint> \
READER_BASE_URL=<Qwen3.5-9B endpoint> JUDGE_BASE_URL=https://api.openai.com/v1 \
OPENAI_API_KEY=<key> python run.py --benchmark lme_v2/small --tag <tag>
```

`--stage evolve` runs the index evolution alone and `--stage evaluate` the six
evaluations of a tag's published keys.

# SELF-INDEX

SELF-INDEX evolves the index of a frozen retriever without relevance labels. Every
document keeps its original text as a key and gains generated keys; a document's score
is its best key's score. With BM25, its generated keys joined into one are also a key.
In each iteration, queries simulated from the corpus are run against the index: the
documents they retrieve are diagnosed, the key sets with a shortfall are revised, and a
new key enters the index only if it passes Faithfulness, Specificity and Separation
checks. The framework runs on retrieval (BRIGHT, table retrieval), search-agent
(BrowseComp-Plus) and agent-memory (LongMemEval-V2) benchmarks.

## Installation

```bash
git clone --recurse-submodules <repository URL>
bash setup.sh --cpu            # or without --cpu on a CUDA machine
source venv/bin/activate
python run.py list
```

`setup.sh` fetches the benchmark code under `externals/` (git submodules), creates
`venv/` with Python 3.12 and installs the dependencies of the BRIGHT and table
experiments. BM25 needs a Java 21 runtime (`JAVA_HOME` or `java` on the
path). The other benchmarks need an optional dependency group:

```bash
python -m pip install -e ".[longmem]"    # LongMemEval-V2
python -m pip install -e ".[browsecomp]" # BrowseComp-Plus
```

`uv.lock` pins every dependency (`uv sync --extra browsecomp`, and so on). Model
weights and datasets are downloaded separately.

## Serving models

The framework talks to OpenAI-compatible endpoints. For example, the optimizer with
vLLM:

```bash
vllm serve Qwen/Qwen3.6-35B-A3B --served-model-name qwen3.6-35b-a3b --port 8000 \
    --max-model-len 262144
export LLM_BASE_URL=http://localhost:8000/v1
```

| Variable | Service |
|---|---|
| `LLM_BASE_URL` | optimizer (index evolution); several replicas comma-separated |
| `EMBEDDING_BASE_URL` | LongMemEval-V2 memory encoder (Qwen3-Embedding-8B) |
| `AGENT_BASE_URL` | BrowseComp-Plus search agent: the GPT-OSS-120B replicas (comma-separated), `https://api.openai.com/v1` for GPT-5.4-nano, `https://openrouter.ai/api/v1` for Kimi-K2.5 |
| `JUDGE_BASE_URL` | BrowseComp-Plus judge (Qwen3-32B) and LongMemEval-V2 evaluator (gpt-5.2) |
| `READER_BASE_URL` | LongMemEval-V2 reader and memory controller |
| `OPENAI_API_KEY` | key sent to every endpoint |
| `GEMINI_API_KEY` | Gemini-3.7-Flash agent |
| `OPENROUTER_API_KEY` | Kimi-K2.5 agent through OpenRouter |

`.env.example` lists them (`set -a; source .env; set +a`).

## Running experiments

```bash
# BRIGHT, one domain, with the BGE, BM25 or Qwen3-Embedding retriever
python run.py --benchmark bright/biology --retriever bge

# Table retrieval, read from data/table_bench/processed/<beaver|fiben|spider2>
python run.py --benchmark table/spider2 --retriever bge

# BrowseComp-Plus: index evolution, agent runs and judging
python run.py --benchmark bcplus --retriever qwen3 --agent-model gptoss

# LongMemEval-V2: index evolution of the memory entries and reader evaluation
python run.py --benchmark lme_v2/small
```

`--retriever bm25` ranks by the max over the original, each key and the joined key;
`--retriever bm25-concat` by max(original, joined keys).

A BRIGHT or table run evolves the index and scores the evaluation queries on the
original index and after the last iteration; the score is the log's `[EVAL]` line and
`summary.json` under `runs/`. BrowseComp-Plus and LongMemEval-V2 also run the two
halves separately: `--stage evolve` evolves the index and publishes its keys, and
`--stage evaluate` evaluates a tag's published keys. `--tag` names a run and
`--dry-run` prints the steps without running them. [docs/REPRODUCING.md](docs/REPRODUCING.md)
gives the commands for the reported numbers and [docs/ARTIFACTS.md](docs/ARTIFACTS.md)
describes the outputs.

## Parameters

| Variable | Meaning | Default |
|---|---|---|
| `ROUNDS` | iterations | 20 |
| `QUERIES_PER_ROUND` | simulated queries per iteration (B) | 128 |
| `RETRIEVAL_DEPTH` | keys retrieved per query (K) | 30 |
| `MAX_KEYS` | keys per document, its original text included (m_max) | 10 |
| `SEED` | random seed | 42 |

`DEVICES` sets the torch devices of the encoders, the index and the searchers
(comma-separated, `cuda:0` by default; `cpu` on a machine without a GPU). With several
devices, the generated keys are split evenly over them; the encoders use the first.

## Extending

Benchmarks, retrievers and commands are registered in code: `register_benchmark` and
`register_domains` (`src/selfindex/runtime/catalog/benchmarks.py`),
`register_retriever` (`src/selfindex/runtime/catalog/retrievers.py`),
`register_factory` (`src/selfindex/runtime/factories.py`) and `register_command`
(`src/selfindex/runtime/commands.py`). A retrieval benchmark subclasses
`RetrievalBenchmark` and implements `load_corpus()`; an encoder implements
`TextEncoder`; a search engine subclasses `RetrieverBackend`.

Each model is called through a backbone of its role in `src/selfindex/llm/backbones.py`
(`OptimizerBackbone`, `AgentBackbone`, `JudgeBackbone`), chosen by the model id. A
model whose requests differ gets a subclass:

```python
@OptimizerBackbone.register("my-org/my-model")
class MyModel(OptimizerBackbone):
    reasoning_effort = "low"
    extra_body = {}

    def text(self, message):
        content = super().text(message)
        return content and content.split("</think>")[-1].strip()
```

## Repository layout

```
src/selfindex/        the framework: index evolution, retrievers, model clients
adapters/browsecomp/  BrowseComp-Plus search agents, indexes and judging
adapters/longmem/     LongMemEval-V2 memory systems and evaluation
externals/            official benchmark code, git submodules at pinned commits (externals/README.md)
runs/                 BRIGHT and table run outputs
docs/                 reproduction commands and run outputs
```

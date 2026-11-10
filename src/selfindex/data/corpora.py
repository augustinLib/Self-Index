from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

from selfindex.runtime.platform.environment import REPO_ROOT


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as load

    return load(*args, **kwargs)


@dataclass
class Corpus:
    name: str
    ids: list
    texts: list
    id2idx: dict
    queries: list
    encoder_settings: dict = field(default_factory=dict)


def _aslist(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            value = ast.literal_eval(x)
            return value if isinstance(value, list) else [x]
        except Exception:
            return [x]
    return list(x)


def load_bright(domain: str) -> Corpus:
    docs = load_dataset("xlangai/BRIGHT", "documents", split=domain)
    examples = load_dataset("xlangai/BRIGHT", "examples", split=domain)
    ids = [str(x) for x in docs["id"]]
    texts = [t or "" for t in docs["content"]]
    id2idx = {u: i for i, u in enumerate(ids)}
    queries = []
    for row in examples:
        gold = _known_rows(row["gold_ids"], id2idx)
        excluded = _known_rows(row["excluded_ids"], id2idx)
        if gold:
            queries.append(
                {
                    "qid": str(row["id"]),
                    "query": row["query"],
                    "gold_idxs": gold,
                    "excluded_idxs": excluded,
                }
            )
    return Corpus(name=domain, ids=ids, texts=texts, id2idx=id2idx, queries=queries)


def _known_rows(document_ids, id2idx):
    return [id2idx[g] for g in _aslist(document_ids) if g in id2idx]


BRIGHT_DOMAINS = (
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "sustainable_living",
    "robotics",
    "stackoverflow",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
)

TABLE_DATASETS = ("beaver", "fiben", "spider2")
_TABLE_ROOT = str(Path(REPO_ROOT) / "data" / "table_bench" / "processed")


def table_root(dataset: str) -> Path:
    return Path(_TABLE_ROOT) / dataset


def load_table(ds: str) -> Corpus:

    root = str(table_root(ds))
    ids, texts = [], []
    with open(f"{root}/tables.jsonl") as f:
        for line in f:
            record = json.loads(line)
            ids.append(str(record["table_id"]))
            texts.append(record.get("content") or "")
    id2idx = {u: i for i, u in enumerate(ids)}
    gold = _table_gold(f"{root}/qrels.tsv", id2idx)
    queries = []
    with open(f"{root}/queries.jsonl") as f:
        for line in f:
            record = json.loads(line)
            qid = str(record["query_id"])
            if record.get("split") == "test" and (judged := gold.get(qid)):
                queries.append(
                    {
                        "qid": qid,
                        "query": record["question"],
                        "gold_idxs": judged,
                        "excluded_idxs": [],
                    }
                )
    return Corpus(name=ds, ids=ids, texts=texts, id2idx=id2idx, queries=queries)


def _table_gold(path, id2idx):
    gold = {}
    with open(path) as f:
        next(f)
        for line in f:
            qid, table_id, relevance = line.rstrip("\n").split("\t")
            if int(relevance) > 0 and table_id in id2idx:
                gold.setdefault(qid, []).append(id2idx[table_id])
    return gold


def load_bcplus() -> Corpus:
    dataset = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    ids = [str(x) for x in dataset["docid"]]
    texts = list(dataset["text"])
    return Corpus(
        name="bcplus",
        ids=ids,
        texts=texts,
        id2idx={i: n for n, i in enumerate(ids)},
        queries=[],
    )

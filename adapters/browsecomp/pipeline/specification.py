import json
import re
from dataclasses import dataclass
from pathlib import Path

from browsecomp.adapters.search_factory import SEARCH_ADAPTERS, get_search_adapter
from selfindex.runtime.platform import environment

REPO_ROOT = Path(environment.REPO_ROOT)
BROWSECOMP_ROOT = Path(__file__).resolve().parents[1]
TOPICS_DIR = BROWSECOMP_ROOT / "topics-qrels"
DATA_DIR = BROWSECOMP_ROOT / "data"


def input_files():
    return {
        "queries": TOPICS_DIR / "queries.tsv",
        "ground_truth": DATA_DIR / "browsecomp_plus_decrypted.jsonl",
        "qrel_evidence": TOPICS_DIR / "qrel_evidence.txt",
    }


def query_ids(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [line.rstrip("\n").split("\t", 1) for line in stream]
    if not rows or any(len(row) != 2 or not row[0] for row in rows):
        raise ValueError("queries must be nonempty query-id/tab/text rows")
    ids = {row[0] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("query IDs must be unique")
    return ids


def verify_evaluation(directory, expected_ids):
    summaries = list(Path(directory).rglob("evaluation_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError("judge must produce exactly one evaluation summary")
    with summaries[0].open(encoding="utf-8") as stream:
        rows = json.load(stream).get("per_query_metrics", [])
    actual_ids = {str(row["query_id"]) for row in rows}
    if actual_ids != expected_ids or len(rows) != len(expected_ids):
        raise RuntimeError("judge results do not cover the requested query IDs")


def validate_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError("BrowseComp tag must be a simple experiment name")
    return tag


def _evolution(tag, adapter, keys):
    evolution = [tag, "--retriever", adapter.evolution_retriever]
    return (
        "bc.pipeline",
        [*evolution, "--keys-out", str(keys.directory / "keys.jsonl")],
    )


def evolution_steps(tag, retriever, keys):
    adapter = get_search_adapter(retriever)
    return (_evolution(tag, adapter, keys), *adapter.preparation_steps(keys))


def evolution_stage_steps(tag, retriever):
    adapter = get_search_adapter(retriever)
    steps = [_evolution(tag, adapter, adapter.key_set(tag))]
    for other in SEARCH_ADAPTERS.values():
        if other.evolution_retriever == adapter.evolution_retriever:
            steps.extend(other.preparation_steps(other.key_set(tag)))
    return tuple(steps)


def agent_arguments(agent, retriever, keys):
    return ["--backbone", agent, "--retriever", retriever, *keys.arguments()]


@dataclass(frozen=True)
class FullRunPlan:
    root: Path
    tag: str
    retriever: str
    agent: str

    @classmethod
    def create(cls, tag, retriever, agent):
        from browsecomp.adapters.paths import run_root

        validate_tag(tag)
        get_search_adapter(retriever).preparation_steps()
        return cls(run_root() / f"full_{tag}", tag, retriever, agent)

    @property
    def keys(self):
        return get_search_adapter(self.retriever).workspace_key_set(self.root)

    def steps(self):
        inputs = input_files()
        runs = str(self.root / "agent")
        return (
            *evolution_steps(self.tag, self.retriever, self.keys),
            (
                "bc.agent",
                [
                    *agent_arguments(self.agent, self.retriever, self.keys),
                    "--out",
                    runs,
                    "--no-judge",
                ],
            ),
            (
                "bc.judge_run",
                [
                    "--input_dir",
                    runs,
                    "--eval_dir",
                    str(self.root / "evals"),
                    "--ground_truth",
                    str(inputs["ground_truth"]),
                    "--qrel_evidence",
                    str(inputs["qrel_evidence"]),
                ],
            ),
        )

import argparse
import os
import time
from dataclasses import dataclass

from browsecomp.adapters.agent_runs import answered_queries, discard_unanswered_runs
from browsecomp.adapters.paths import backbone_run_dir
from browsecomp.adapters.search_factory import add_search_arguments, key_set_from
from browsecomp.agent._paths import BROWSECOMP_DIR
from browsecomp.agent.backbones import agent_names, get_backbone, judge_and_report
from browsecomp.costs.usage import print_usage
from browsecomp.pipeline.specification import input_files, query_ids

MAX_AGENT_THREADS = 400
MAX_PASSES = 5
RETRY_PASS_DELAY_SECONDS = 60


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=agent_names())
    add_search_arguments(parser)
    parser.add_argument(
        "--out",
        help="run directory (default: runs/<key set or base>/<backbone>-<retriever> "
        "under adapters/browsecomp)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="leave judging to a later step of the caller",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True)
class _AgentRun:
    queries: str
    query_ids: frozenset
    out: str
    threads: int
    log_dir: str
    log: str


def _prepare_run(options, keys):
    queries = input_files()["queries"]
    expected = frozenset(query_ids(queries))
    threads = min(MAX_AGENT_THREADS, len(expected))
    out = options.out or backbone_run_dir(options.backbone, options.retriever, keys)
    log_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.abspath(out) + ".log"
    print(
        f"[{options.backbone}] {options.retriever} key={int(keys is not None)} · "
        f"{len(expected)} queries · concurrency {threads} · out={out} · log={log}",
        flush=True,
    )
    return _AgentRun(str(queries), expected, out, threads, log_dir, log)


def answer_every_query(backbone, run):
    for attempt in range(1, MAX_PASSES + 1):
        discarded = discard_unanswered_runs(run.out)
        missing = run.query_ids - answered_queries(run.out)
        if not missing:
            break
        if attempt > 1:
            time.sleep(RETRY_PASS_DELAY_SECONDS)
        note = (
            f", {discarded} runs without a model answer discarded" if discarded else ""
        )
        print(
            f"[{backbone.name}] pass {attempt}/{MAX_PASSES}: {len(missing)} of "
            f"{len(run.query_ids)} queries unanswered{note}",
            flush=True,
        )
        returncode = backbone.run_agent(run.queries, run.out, run.threads, run.log)
        newly_answered = len(missing & answered_queries(run.out))
        print(
            f"[{backbone.name}] pass {attempt} exit {returncode}, "
            f"{newly_answered} answered",
            flush=True,
        )
        if returncode and not newly_answered:
            raise RuntimeError(
                f"[{backbone.name}] agent failed (exit {returncode}); see {run.log}"
            )
    answered = answered_queries(run.out)
    if missing := run.query_ids - answered:
        raise RuntimeError(
            f"[{backbone.name}] {len(missing)} of {len(run.query_ids)} queries are "
            f"unanswered after {MAX_PASSES} passes; see {run.log}"
        )
    if unexpected := answered - run.query_ids:
        raise RuntimeError(
            f"[{backbone.name}] {run.out} holds runs of {len(unexpected)} queries "
            "outside the query set"
        )


def main(argv=None):
    options = _parse_args(argv)
    os.chdir(BROWSECOMP_DIR)
    keys = key_set_from(options)
    backbone = get_backbone(options.backbone, retriever=options.retriever, keys=keys)
    backbone.check_service()
    if not options.no_judge:
        from browsecomp.agent.judge_run import require_judge

        require_judge()
    run = _prepare_run(options, keys)
    answer_every_query(backbone, run)
    print_usage(run.out, backbone.prices)
    if not options.no_judge:
        judge_and_report(backbone, run.out, run.log_dir, run.query_ids)

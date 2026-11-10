from __future__ import annotations

import argparse
import inspect
import json
import os
from abc import ABC, abstractmethod

from selfindex.common.io import read_json
from selfindex.runtime.platform import environment
from selfindex.runtime.registry import Registry


class Benchmark(ABC):

    name = ""
    family = ""
    description = ""

    @abstractmethod
    def build_experiment(self, spec) -> list: ...

    def validate_spec(self, spec): ...

    def describe(self) -> str:
        return self.description or self.name


class RetrievalBenchmark(Benchmark):

    family = "retrieval"
    collection = ""
    default_retriever = "bge"

    def __init__(self, domain: str):
        if (
            not isinstance(domain, str)
            or not domain.strip()
            or any(mark in domain for mark in ("/", "\\"))
            or domain in (".", "..")
        ):
            raise ValueError("a domain must be one nonempty path component")
        self.domain = domain
        self.name = f"{self.collection}/{domain}"

    @abstractmethod
    def load_corpus(self): ...

    def runs_directory(self):
        return environment.RETRIEVAL_RUNS_ROOT

    def validate_spec(self, spec):
        if spec.agent_model:
            raise ValueError(f"{self.name} has no downstream --agent-model")
        if spec.stage == "evolve":
            raise ValueError(
                f"{self.name} scores the index in the same run: "
                "use --stage full, or --stage evaluate for a finished run"
            )

    def score(self, run_ref):
        path = os.path.join(
            self.runs_directory(), f"{self.domain}_pipe_{run_ref}", "summary.json"
        )
        summary = read_json(path)
        return {
            "base": summary.get("base"),
            "evolved": summary.get("evolved"),
            "retriever": summary.get("retriever"),
        }

    def build_experiment(self, spec):
        tag = spec.tag or self.domain
        if spec.stage == "evaluate":
            return [(self.family, "result", [self.name, tag], {})]
        retriever = spec.retriever or self.default_retriever
        return [(self.family, "run", [self.domain, tag, "--retriever", retriever], {})]


class BrightBenchmark(RetrievalBenchmark):
    description = "BRIGHT domain: retrieval nDCG@10 over a frozen test split."

    collection = "bright"

    def load_corpus(self):
        from selfindex.data.corpora import load_bright

        return load_bright(self.domain)


class TableBenchmark(RetrievalBenchmark):
    description = (
        "Table retrieval (fiben / spider2 / beaver): nDCG@10 over the test split."
    )

    collection = "table"

    def load_corpus(self):
        from selfindex.data.corpora import load_table

        return load_table(self.domain)


class BCPlusBenchmark(Benchmark):
    description = (
        "BrowseComp-Plus: 830-query search-agent task, LLM-judged accuracy/recall."
    )

    name, family = "bcplus", "browsecomp"
    DEFAULT_TAG = "selfindex"
    DEFAULT_AGENT = "gptoss"

    def validate_spec(self, spec):
        if spec.stage == "evolve" and spec.agent_model:
            raise ValueError("--agent-model is only used during BrowseComp evaluation")
        self._search_adapter(spec)
        self._tag(spec)
        if spec.stage != "evolve":
            self._agent(spec)

    @staticmethod
    def _search_adapter(spec):
        from browsecomp.adapters.search_factory import search_adapter_name
        from selfindex.runtime.catalog.retrievers import RETRIEVERS

        retriever = RETRIEVERS[spec.retriever] if spec.retriever else None
        return search_adapter_name(retriever)

    def _tag(self, spec):
        from browsecomp.adapters.paths import BASE_INDEX_RUNS
        from browsecomp.pipeline.specification import validate_tag

        tag = validate_tag(spec.tag or self.DEFAULT_TAG)
        if spec.stage != "full" and tag == BASE_INDEX_RUNS:
            raise ValueError(f"tag {tag} names the base-index runs")
        return tag

    def _agent(self, spec):
        from browsecomp.agent.backbones import agent_names

        agent = spec.agent_model or self.DEFAULT_AGENT
        if agent not in agent_names():
            raise ValueError(
                f"unsupported BrowseComp agent: {agent} "
                f"(agents: {', '.join(agent_names())})"
            )
        return agent

    def load_corpus(self):
        from selfindex.data.corpora import load_bcplus

        return load_bcplus()

    def runs_directory(self):
        from browsecomp.adapters.paths import run_root

        return run_root()

    def build_experiment(self, spec):
        from browsecomp.adapters.search_factory import get_search_adapter

        retriever, tag = self._search_adapter(spec), self._tag(spec)
        if spec.stage == "full":
            return [self._full_run_step(spec, retriever, tag)]
        if spec.stage == "evolve":
            return self._evolution_steps(retriever, tag)
        from browsecomp.pipeline.specification import agent_arguments

        keys = get_search_adapter(retriever).key_set(tag)
        argv = agent_arguments(self._agent(spec), retriever, keys)
        return [(self.family, "agent", argv, {})]

    def _evolution_steps(self, retriever, tag):
        from browsecomp.pipeline.specification import evolution_stage_steps
        from selfindex.runtime.factories import BrowseCompFactory

        tasks = {command: task for task, command in BrowseCompFactory.commands.items()}
        steps = []
        for command, argv in evolution_stage_steps(tag, retriever):
            if command not in tasks:
                raise ValueError(f"{command} is not a BrowseComp task")
            steps.append((self.family, tasks[command], argv, {}))
        return steps

    def _full_run_step(self, spec, retriever, tag):
        from browsecomp.pipeline.full_run import FullRun

        agent = self._agent(spec)
        FullRun.create(tag, retriever, agent)
        argv = [tag, "--retriever", retriever, "--agent", agent]
        return (self.family, "full", argv, {})


class LongMemBenchmark(RetrievalBenchmark):
    description = "LongMemEval-V2 Small: reader QA over the evolved memory index, judged accuracy."

    family, collection = "longmem", "lme_v2"

    def __init__(self):
        super().__init__("small")
        self.domain = "lme_v2_small"

    def load_corpus(self):
        from longmem.adapter.corpus import memory_corpus

        return memory_corpus(self.domain)

    def runs_directory(self):
        from longmem.adapter.paths import OUTPUT_ROOT

        return OUTPUT_ROOT

    def validate_spec(self, spec):
        from longmem.adapter.retrieval import require_no_retriever

        require_no_retriever(spec.retriever)
        if spec.agent_model:
            raise ValueError(
                "the LongMemEval-V2 reader is the model READER_BASE_URL serves; "
                "omit --agent-model"
            )

    def build_experiment(self, spec):
        from longmem.adapter.harness_options import DOMAINS, MEMORY_TEMPLATES
        from longmem.adapter.paths import evaluation_dir, keys_path
        from longmem.adapter.retrieval import EVOLUTION_RETRIEVER

        tag = spec.tag or "selfindex"
        keys = str(keys_path(tag))
        steps = []
        if spec.stage != "evaluate":
            argv = [self.domain, tag, "--retriever", EVOLUTION_RETRIEVER]
            steps.append((self.family, "evolve", [*argv, "--keys-out", keys], {}))
        if spec.stage == "evolve":
            return steps
        for memory in MEMORY_TEMPLATES:
            for domain in DOMAINS:
                out = str(evaluation_dir(tag, memory, domain))
                steps.append((self.family, "eval", [domain, out, keys, memory], {}))
        return steps


BENCHMARKS: Registry[Benchmark] = Registry("benchmark")
CORPORA: Registry[Benchmark] = Registry("pipeline corpus")


def register_benchmark(benchmark: Benchmark):
    if not isinstance(benchmark, Benchmark):
        raise TypeError("benchmark must implement Benchmark")
    retrieval = isinstance(benchmark, RetrievalBenchmark)
    BENCHMARKS.validate_name(benchmark.name)
    if retrieval:
        CORPORA.validate_name(benchmark.domain)
    BENCHMARKS.register(benchmark.name, benchmark)
    if retrieval:
        CORPORA.register(benchmark.domain, benchmark)
    return benchmark


def register_domains(benchmark_type, domains):
    if (
        not isinstance(benchmark_type, type)
        or not issubclass(benchmark_type, RetrievalBenchmark)
        or inspect.isabstract(benchmark_type)
    ):
        raise TypeError("domains require a concrete RetrievalBenchmark subclass")
    if isinstance(domains, str):
        raise TypeError("domains must be an iterable of domain names, not a string")
    adapters = tuple(benchmark_type(domain) for domain in domains)
    names, corpora = set(), set()
    for adapter in adapters:
        BENCHMARKS.validate_name(adapter.name)
        CORPORA.validate_name(adapter.domain)
        if adapter.name in names or adapter.domain in corpora:
            raise ValueError("duplicate domain registration")
        names.add(adapter.name)
        corpora.add(adapter.domain)
    for adapter in adapters:
        register_benchmark(adapter)
    return adapters


def _corpus_owner(name):
    if name not in CORPORA:
        raise ValueError(
            f"unknown corpus: {name} (registered: {', '.join(sorted(CORPORA))})"
        )
    return CORPORA[name]


def load_corpus(name):
    return _corpus_owner(name).load_corpus()


def runs_directory(name):
    return _corpus_owner(name).runs_directory()


def score_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print a finished retrieval run's scores"
    )
    parser.add_argument(
        "benchmark",
        choices=sorted(
            name
            for name, benchmark in BENCHMARKS.items()
            if benchmark.family == RetrievalBenchmark.family
        ),
    )
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    report = BENCHMARKS[args.benchmark].score(args.tag)
    report_json = json.dumps(report, indent=1)
    print(report_json)


def _build_registry():
    from selfindex.data.corpora import BRIGHT_DOMAINS, TABLE_DATASETS

    register_domains(BrightBenchmark, BRIGHT_DOMAINS)
    register_domains(TableBenchmark, TABLE_DATASETS)
    CORPORA.register("bcplus", register_benchmark(BCPlusBenchmark()))
    register_benchmark(LongMemBenchmark())


_build_registry()

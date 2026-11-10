from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from collections.abc import Mapping
from typing import TYPE_CHECKING

from selfindex.runtime.workflows.execution import BenchmarkWorkflow, Command, Stage
from selfindex.runtime.workflows.checkpoints import (
    configuration_environment,
    file_digest,
    identity_digest,
    source_digests,
)

if TYPE_CHECKING:
    from browsecomp.pipeline.specification import FullRunPlan


@dataclass(frozen=True)
class BrowseCompWorkflow(BenchmarkWorkflow):
    run: "FullRunPlan"
    environment: Mapping[str, str]
    resumed: bool = False

    def __post_init__(self):
        object.__setattr__(
            self, "environment", MappingProxyType(dict(self.environment))
        )

    @property
    def output_directory(self):
        return self.run.root

    @property
    def resume(self):
        return self.resumed

    @property
    def execution_identity(self):
        from browsecomp.pipeline.specification import input_files, REPO_ROOT

        packages = ("src/selfindex", "adapters/browsecomp")
        environment = configuration_environment(self.environment)
        value = {
            "tag": self.run.tag,
            "retriever": self.run.retriever,
            "agent": self.run.agent,
            "environment": environment,
            "sources": source_digests(REPO_ROOT, packages),
            "inputs": {name: file_digest(path) for name, path in input_files().items()},
        }
        return identity_digest(value)

    def prepare(self):
        from browsecomp.pipeline.specification import input_files, query_ids

        inputs = input_files()
        for path in inputs.values():
            if not path.is_file():
                raise FileNotFoundError(f"required BrowseComp input: {path}")
        query_ids(inputs["queries"])
        if self.resume:
            if not (self.run.root / "workflow_receipts.json").is_file():
                raise ValueError("resume requires workflow receipts")
        else:
            self.run.root.mkdir(parents=True, exist_ok=False)

    def _artifacts(self, command):
        root, index = self.run.root, self.run.keys.index
        locations = {
            "bc.pipeline": (root / "keys/keys.jsonl",),
            "bc.encode_keys": (root / "keys/key_emb.npz",),
            "bc.build_index": tuple(index.rglob("*")) if index is not None else (),
            "bc.agent": tuple((root / "agent").rglob("run_*.json")),
            "bc.judge_run": tuple((root / "evals").rglob("*.json")),
        }
        return tuple(path for path in locations.get(command, ()) if path.is_file())

    def stages(self):
        from browsecomp.pipeline.specification import (
            BROWSECOMP_ROOT,
            input_files,
            query_ids,
            verify_evaluation,
        )

        environment = self.environment
        commands = self.run.steps()
        expected = query_ids(input_files()["queries"])
        stages = tuple(
            Stage(
                f"{index}:{name}",
                Command(name, tuple(arguments), environment, BROWSECOMP_ROOT),
                artifacts=partial(self._artifacts, name),
            )
            for index, (name, arguments) in enumerate(commands)
        )
        return stages + (
            Stage(
                "verify-evaluation",
                partial(verify_evaluation, self.run.root / "evals", expected),
            ),
        )

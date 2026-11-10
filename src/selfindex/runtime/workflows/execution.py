import fcntl
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from selfindex.runtime.commands import subprocess_argv
from selfindex.runtime.workflows.checkpoints import WorkflowCheckpoint
from selfindex.storage.artifacts import LocalArtifactStore


@dataclass(frozen=True)
class Stage:
    name: str
    action: Callable[[], int | None]
    artifacts: Callable[[], tuple[Path, ...]] | None = None


@dataclass(frozen=True)
class Command:
    name: str
    arguments: tuple[str, ...]
    environment: Mapping[str, str]
    directory: Path

    def __call__(self) -> int:
        result = subprocess.run(
            subprocess_argv(self.name, list(self.arguments)),
            env=dict(self.environment),
            cwd=self.directory,
        )
        return result.returncode


class StageRunner:
    def __init__(self, directory: Path, *, identity, resume):
        self.store = LocalArtifactStore(directory)
        self.identity = identity
        self.resume = resume

    def run(self, stages: tuple[Stage, ...]) -> int:
        names = [stage.name for stage in stages]
        if len(names) != len(set(names)):
            raise ValueError("workflow stage names must be unique")
        self.store.root.mkdir(parents=True, exist_ok=True)
        with (self.store.root / "workflow.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checkpoint = WorkflowCheckpoint(
                self.store.root, self.identity, names, self.resume
            )
            return self._execute(stages, checkpoint)

    def _execute(self, stages, checkpoint):
        completed = []
        for stage in stages:
            if checkpoint.complete(stage.name):
                completed.append(stage.name)
                continue
            self._record(stage.name, "running", completed)
            try:
                code = self._run_stage(stage, checkpoint, completed)
            except BaseException as error:
                self._record(
                    stage.name, "failed", completed, error_type=type(error).__name__
                )
                raise
            if code is not None:
                return code
            completed.append(stage.name)
            self._record(stage.name, "complete", completed)
        self._record(None, "complete", completed)
        return 0

    def _run_stage(self, stage, checkpoint, completed):
        code = stage.action()
        if code is not None and type(code) is not int:
            raise TypeError("stage must return an integer exit code or None")
        if code:
            self._record(stage.name, "failed", completed, exit_code=code)
            return code
        if stage.artifacts is not None:
            paths = stage.artifacts()
            if paths:
                checkpoint.commit(stage.name, paths)
        return None

    def _record(self, stage, status, completed, **details):
        record = {
            "stage": stage,
            "status": status,
            "completed": list(completed),
            **details,
        }
        self.store.write_json("workflow_status.json", record)


class BenchmarkWorkflow(ABC):
    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def stages(self) -> tuple[Stage, ...]: ...

    @property
    @abstractmethod
    def output_directory(self) -> Path: ...

    @property
    @abstractmethod
    def execution_identity(self): ...

    @property
    @abstractmethod
    def resume(self): ...

    def execute(self) -> int:
        self.prepare()
        return StageRunner(
            self.output_directory, identity=self.execution_identity, resume=self.resume
        ).run(self.stages())

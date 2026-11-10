from __future__ import annotations

import inspect
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from selfindex.runtime.platform import environment
from selfindex.runtime.registry import Registry


def import_path(root):
    return os.pathsep.join(os.path.join(root, name) for name in ("src", "adapters"))


@dataclass
class Task:
    argv: list
    cwd: str
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        self.env = dict(self.env or {})

    def run(self) -> int:
        env = dict(os.environ)
        env.update(self.env)
        return subprocess.run(self.argv, cwd=self.cwd, env=env).returncode


class BenchmarkFactory(ABC):
    name = ""
    tasks: tuple = ()
    services: dict = {}

    @classmethod
    def validate_task(cls, task):
        if cls.tasks and task not in cls.tasks:
            raise ValueError(
                f"[{cls.name}] unknown task: {task} (tasks: {sorted(cls.tasks)})"
            )

    @classmethod
    def required_services(cls, task):
        return cls.services.get(task, ())

    def __init__(self):
        self.root = environment.REPO_ROOT

    @abstractmethod
    def build(self, task: str, argv: list) -> Task: ...

    def create(self, task, argv):
        self.validate_task(task)
        result = self.build(task, list(argv))
        if not isinstance(result, Task):
            raise TypeError("factory build() must return Task")
        return result

    def _cmd(self, command, args=(), extra_env=None, cwd=None):
        from selfindex.runtime.commands import subprocess_argv

        return Task(
            subprocess_argv(command, args), cwd=cwd or self.root, env=extra_env or {}
        )


class RetrievalFactory(BenchmarkFactory):

    name = "retrieval"
    services = {"run": ("LLM",)}
    tasks = ("run", "result")

    def build(self, task, argv):
        env = {"PYTHONPATH": import_path(self.root)}
        if task == "result":
            return self._cmd("retrieval.result", argv, extra_env=env)
        if len(argv) != 4 or argv[2] != "--retriever":
            raise ValueError("a retrieval run takes <domain> <tag> --retriever <name>")
        return LaunchTask(self.root, argv, env)


class LaunchTask(Task):

    def __init__(self, root, argv, env):
        from selfindex.runtime.commands import subprocess_argv

        super().__init__(
            argv=subprocess_argv("pipeline.launch", argv), cwd=root, env=env
        )
        self.domain, self.tag = argv[0], argv[1]

    def run(self):
        env = {**os.environ, **self.env}
        log = os.path.join(environment.RETRIEVAL_RUNS_ROOT, f"run_{self.tag}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if os.path.exists(log):
            os.rename(log, log + "." + time.strftime("%m%d_%H%M%S"))
        with open(log, "w") as fh:
            process = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(
            f"[launch] dom={self.domain} tag={self.tag} pid={process.pid}, "
            f"log: tail -f {log}",
            flush=True,
        )
        return 0


class BrowseCompFactory(BenchmarkFactory):

    name = "browsecomp"
    services = {"pipeline": ("LLM",), "agent": ("JUDGE",), "full": ("LLM", "JUDGE")}
    tasks = ("pipeline", "encode_keys", "build_index", "agent", "full")
    commands = {
        "pipeline": "bc.pipeline",
        "encode_keys": "bc.encode_keys",
        "build_index": "bc.build_index",
        "agent": "bc.agent",
        "full": "bc.full",
    }

    def build(self, task, argv):
        return self._cmd(
            self.commands[task],
            argv,
            cwd=os.path.join(self.root, "adapters", "browsecomp"),
        )


class LongMemFactory(BenchmarkFactory):

    name = "longmem"
    services = {"evolve": ("LLM",), "eval": ("READER", "EMBEDDING", "JUDGE")}
    tasks = ("evolve", "eval")
    commands = {"evolve": "pipeline.launch", "eval": "lme.eval"}

    def build(self, task, argv):
        return self._cmd(
            self.commands[task], argv, extra_env={"PYTHONPATH": import_path(self.root)}
        )


FACTORIES: Registry[type[BenchmarkFactory]] = Registry("benchmark factory")


def register_factory(factory):

    if (
        not isinstance(factory, type)
        or not issubclass(factory, BenchmarkFactory)
        or inspect.isabstract(factory)
    ):
        raise TypeError("factory must be a concrete BenchmarkFactory subclass")
    return FACTORIES.register(factory.name, factory)


for _factory in (RetrievalFactory, BrowseCompFactory, LongMemFactory):
    register_factory(_factory)

from __future__ import annotations

from dataclasses import dataclass, field

from selfindex.runtime.catalog.retrievers import RETRIEVERS
from selfindex.runtime.configuration import validate_services
from selfindex.runtime.factories import FACTORIES
from selfindex.runtime.plan import ExecutionPlan, StepSpec


@dataclass(frozen=True)
class ExperimentSpec:

    benchmark: str
    retriever: str | None = None
    agent_model: str | None = None
    stage: str = "full"
    tag: str | None = None
    runtime_env: dict = field(default_factory=dict)


def _validate_options(spec, benchmarks):
    if spec.benchmark not in benchmarks:
        raise ValueError(f"unknown benchmark: {spec.benchmark} (see run.py list)")
    if spec.retriever is not None and spec.retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever: {spec.retriever}")
    if spec.stage not in ("evolve", "evaluate", "full"):
        raise ValueError(f"unknown stage: {spec.stage}")


def _plan_step(spec, family, task, argv, env):
    if family not in FACTORIES:
        raise ValueError(f"unknown benchmark factory: {family}")
    FACTORIES[family].validate_task(task)
    env = {**env, **spec.runtime_env}
    return StepSpec(family, task, list(argv), env)


class Runner:

    def plan(self, spec: ExperimentSpec) -> ExecutionPlan:
        from selfindex.runtime.catalog.benchmarks import BENCHMARKS

        _validate_options(spec, BENCHMARKS)
        bench = BENCHMARKS[spec.benchmark]
        bench.validate_spec(spec)
        steps = tuple(
            _plan_step(spec, family, task, argv, env)
            for family, task, argv, env in bench.build_experiment(spec)
        )
        return ExecutionPlan(spec.benchmark, steps)

    def run_experiment(self, spec: ExperimentSpec) -> int:
        plan = self.plan(spec)

        validate_services(plan)
        tasks = self._build_tasks(plan)
        for step, task in zip(plan.steps, tasks, strict=True):
            print(
                f"[experiment] {spec.benchmark} -> "
                f"{step.family} {step.task} {step.argv}",
                flush=True,
            )
            if (code := task.run()) != 0:
                return code
        return 0

    def _build_tasks(self, plan):
        tasks = []
        for family, task, argv, env in plan.steps:
            built = FACTORIES[family]().create(task, argv)
            built.env.update(env)
            tasks.append(built)
        return tasks

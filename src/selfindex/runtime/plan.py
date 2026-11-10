from dataclasses import dataclass
from typing import NamedTuple


class StepSpec(NamedTuple):
    family: str
    task: str
    argv: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class ExecutionPlan:
    benchmark: str
    steps: tuple[StepSpec, ...]

    def to_dict(self):
        return {"benchmark": self.benchmark, "steps": [s._asdict() for s in self.steps]}

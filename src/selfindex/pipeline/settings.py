import os
from dataclasses import dataclass

PARAMETERS = {
    "ROUNDS": 20,
    "QUERIES_PER_ROUND": 128,
    "RETRIEVAL_DEPTH": 30,
    "MAX_KEYS": 10,
    "SEED": 42,
}
POSITIVE_PARAMETERS = ("QUERIES_PER_ROUND", "RETRIEVAL_DEPTH", "MAX_KEYS")


def parameter(name) -> int:
    value = int(os.environ.get(name, PARAMETERS[name]))
    if name in POSITIVE_PARAMETERS and value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class LoopSettings:
    rounds: int = PARAMETERS["ROUNDS"]

    @classmethod
    def from_env(cls):
        return cls(rounds=max(1, parameter("ROUNDS")))


@dataclass(frozen=True)
class SignalSettings:
    queries_per_round: int = PARAMETERS["QUERIES_PER_ROUND"]
    retrieval_depth: int = PARAMETERS["RETRIEVAL_DEPTH"]
    seed: int = PARAMETERS["SEED"]

    @classmethod
    def from_env(cls):
        return cls(
            queries_per_round=parameter("QUERIES_PER_ROUND"),
            retrieval_depth=parameter("RETRIEVAL_DEPTH"),
            seed=parameter("SEED"),
        )


@dataclass(frozen=True)
class PipelineSettings:
    retriever: str
    rounds: int = PARAMETERS["ROUNDS"]

    @classmethod
    def from_env(cls, retriever):
        return cls(retriever=retriever, rounds=parameter("ROUNDS"))

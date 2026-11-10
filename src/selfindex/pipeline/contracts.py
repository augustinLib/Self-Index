from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    import numpy as np

    from selfindex.index.contracts import KeyIndex
    from selfindex.pipeline.exploration.simulator import QueryGenerator
    from selfindex.retrieval.contracts import Retriever


class TextEncoding(Protocol):
    def __call__(self, texts: Sequence[str], **options) -> np.ndarray: ...


class GenerationClient(Protocol):
    def batch(self, messages: list, **options) -> list[str]: ...


@dataclass
class RoundState:

    round: int = 0
    phase_seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class SelectionState:
    information: np.ndarray | None = None
    units: set[int] | None = None


@dataclass
class QueryState:
    rounds: dict[int, dict] | None = None

    def used_before(self, round_index: int) -> list[str]:
        used = []
        for round_number in range(round_index):
            if bank := (self.rounds or {}).get(round_number):
                used.extend(bank["Q"])
        return used

    def sources_before(self, round_index: int) -> set[int]:
        sources = set()
        for round_number in range(round_index):
            if bank := (self.rounds or {}).get(round_number):
                sources.update(int(source) for source in bank["SRC"])
        return sources


@dataclass
class Context:

    texts: list
    encode: TextEncoding
    llm: GenerationClient
    index: KeyIndex
    retriever: Retriever
    gen: QueryGenerator
    eligible: list
    run_out: str
    round_state: RoundState = field(default_factory=RoundState)
    query_state: QueryState = field(default_factory=QueryState, repr=False)
    selection_state: SelectionState = field(default_factory=SelectionState, repr=False)


@dataclass
class Signals:

    demand_order: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class EditState:
    processed: int = 0
    committed: int = 0
    revised: int = 0
    keys: int = 0
    edited: set = field(default_factory=set)
    t0: float = 0.0


class _Named:

    name = "base"


class SignalSource(_Named, ABC):

    @abstractmethod
    def build(self, ctx: Context) -> Signals:
        raise NotImplementedError


class DocSelector(_Named, ABC):

    @abstractmethod
    def order(self, ctx: Context, sig: Signals) -> list:
        raise NotImplementedError

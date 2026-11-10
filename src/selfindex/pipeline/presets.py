from typing import NamedTuple

from selfindex.pipeline.contracts import SignalSource
from selfindex.pipeline.strategies.signals import QuerySimulator

PRESET = "self-index"


class StrategySet(NamedTuple):
    signal: SignalSource


def build_slots():
    return StrategySet(QuerySimulator())

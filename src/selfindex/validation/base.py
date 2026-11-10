from abc import ABC, abstractmethod
from dataclasses import dataclass

from selfindex.revision.base import ProposedKey


def _finite_scores(scores):
    import torch

    if not torch.isfinite(scores).all():
        raise ValueError("validation received non-finite scores")
    return scores


class EncodedTexts:

    def __init__(self, live, texts):
        self._rows = {text: row for row, text in enumerate(dict.fromkeys(texts))}
        self._encoded = live.query_repr(list(self._rows))

    def __call__(self, texts):
        rows = [self._rows[text] for text in texts]
        if isinstance(self._encoded, list):
            return [self._encoded[row] for row in rows]
        return self._encoded[rows]


@dataclass(frozen=True)
class KeyChunk:

    texts: tuple
    owners: tuple
    queries: object


class ScoreCriterion(ABC):

    @abstractmethod
    def apply(self, chunk, keep, stats): ...


@dataclass(frozen=True)
class RevisionDecision:
    doc: int
    accepted: tuple[ProposedKey, ...]

    @property
    def replacement(self):
        if not any(key.is_new for key in self.accepted):
            return None
        return tuple(key.text for key in self.accepted)

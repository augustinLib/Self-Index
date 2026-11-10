import math
import re
from collections import Counter

JACCARD_TAU = 0.8


_WORD = re.compile(r"[^\W_]+")


def words(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared)


class Dissimilarity:

    def __init__(self):
        self._sets: list[frozenset] = []
        self._document_frequency: Counter = Counter()
        self._postings: dict[str, list[int]] = {}
        self._frozen = False

    def _prefix(self, tokens: frozenset) -> list[str]:
        rarest_first = sorted(
            tokens, key=lambda token: (self._document_frequency[token], token)
        )
        length = len(rarest_first) - math.ceil(JACCARD_TAU * len(rarest_first)) + 1
        return rarest_first[: max(length, 1)]

    def _index(self, position: int, tokens: frozenset) -> None:
        for token in self._prefix(tokens):
            self._postings.setdefault(token, []).append(position)

    def add(self, text: str) -> None:
        if not (tokens := words(text)):
            return
        position = len(self._sets)
        self._sets.append(tokens)
        if not self._frozen:
            self._document_frequency.update(tokens)
        self._index(position, tokens)

    def is_novel(self, text: str) -> bool:
        if not self._frozen:
            self._frozen = True
            self._postings = {}
            for position, stored in enumerate(self._sets):
                self._index(position, stored)
        tokens = words(text)
        candidates = {
            position
            for token in self._prefix(tokens)
            for position in self._postings.get(token, ())
        }
        return all(
            jaccard(tokens, self._sets[position]) < JACCARD_TAU
            for position in sorted(candidates)
        )

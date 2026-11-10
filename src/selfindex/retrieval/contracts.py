from __future__ import annotations

from abc import ABC, abstractmethod

SCORE_BATCH_SIZE = 64


class Retriever(ABC):
    def snapshot(self, texts):
        raise NotImplementedError(
            f"{type(self).__name__} does not support iteration snapshots"
        )

    @abstractmethod
    def score_queries(self, texts, query_embeddings=None):
        raise NotImplementedError(f"{type(self).__name__} must implement score_queries")

    def score_query_batches(self, texts, query_embeddings=None):
        if not len(texts):
            return
        scores = self.score_queries(texts, query_embeddings)
        for start in range(0, len(scores), SCORE_BATCH_SIZE):
            yield scores[start : start + SCORE_BATCH_SIZE]

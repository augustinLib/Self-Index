import numpy as np

from selfindex.common.fusion import fold_max_np

EVOLUTION_RETRIEVER = "qwen3"


class MaxFusionRetriever:

    def scores(self, query, documents, keys, owners):
        original_scores = np.asarray(documents, np.float32) @ query
        key_scores = keys @ query
        return fold_max_np(original_scores[None, :], key_scores[None, :], owners)[0]

    def rank(self, query, documents, keys, owners, top_k):
        scores = np.asarray(self.scores(query, documents, keys, owners))
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise ValueError("memory scores must be finite, one per document")
        indices = np.argsort(scores)[::-1][: int(top_k)]
        return [(int(index), float(scores[index])) for index in indices]


def require_no_retriever(retriever=None):
    if retriever:
        raise ValueError(
            "LongMemEval-V2 ranks with the memory systems' encoder; omit --retriever"
        )

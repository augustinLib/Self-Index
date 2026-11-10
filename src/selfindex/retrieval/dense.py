from __future__ import annotations

import numpy as np

from selfindex.retrieval.contracts import SCORE_BATCH_SIZE, Retriever


class DenseRetriever(Retriever):

    def __init__(self, index, encode_fn):
        self.ix = index
        self._encode = encode_fn

    def snapshot(self, texts):
        from selfindex.retrieval.snapshots import DenseSnapshot

        return DenseSnapshot(self, texts)

    def score_queries(self, texts, query_embeddings=None):
        scores = None
        start = 0
        for block in self.score_query_batches(texts, query_embeddings):
            if scores is None:
                scores = np.empty((len(texts), self.ix.N), block.dtype)
            scores[start : start + len(block)] = block
            start += len(block)
        return scores if scores is not None else np.empty((0, self.ix.N), np.float32)

    def score_query_batches(self, texts, query_embeddings=None):
        if not len(texts):
            return
        if query_embeddings is None:
            query_embeddings = self._encode(texts, is_query=True)
        embeddings = np.asarray(query_embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
            raise ValueError("query embeddings must align with query texts")
        for start in range(0, len(texts), SCORE_BATCH_SIZE):
            yield self._score_tensor_batch(embeddings[start : start + SCORE_BATCH_SIZE])

    def _score_tensor_batch(self, queries):
        torch = self.ix._torch
        queries = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(
            self.ix._dev
        )
        first = self._score_tensor(queries[0].clone())
        scores = first.new_empty((len(queries), self.ix.N))
        scores[0] = first
        for row in range(1, len(queries)):
            scores[row] = self._score_tensor(queries[row].clone())
        return scores.cpu().numpy()

    def _score_tensor(self, query):
        ix = self.ix
        with ix._lock:
            scores = ix.emb_orig_t @ query
            keys = (
                ix.best_key_scores(query, out_dev=ix._dev)[0] if ix.key_count else None
            )
        return scores if keys is None else ix._torch.maximum(scores, keys)

from functools import cached_property

import numpy as np

from selfindex.index.contracts import ScoringView
from selfindex.runtime.platform.devices import compute_device


class LiveIndex(ScoringView):

    def __init__(self, retriever, doc_texts):
        self.retriever = retriever
        self.ix = retriever.ix
        self.texts = doc_texts

    @property
    def device(self):
        return self.ix.emb_orig_t.device

    @property
    def N(self):
        return int(self.ix.N)

    def query_repr(self, texts):
        import torch

        embeddings = np.asarray(
            self.retriever._encode(list(texts), is_query=True), np.float32
        )
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(texts)
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError("query embeddings must be finite and aligned with texts")
        return torch.from_numpy(np.ascontiguousarray(embeddings)).to(self.device)

    def doc_query_repr(self, docs):
        return self.query_repr([self.texts[int(d)] for d in docs])

    def original_scores(self, queries):
        with self.ix._lock:
            return queries @ self.ix.emb_orig_t.T

    def key_scores(self, qt, kids):
        key_vectors = self.ix.kid_rows(list(kids), dev=qt.device)
        return qt @ key_vectors.T

    def document_scores(self, queries):
        import torch

        with self.ix._lock:
            scores = queries @ self.ix.emb_orig_t.T
            return torch.maximum(
                scores, self.ix.best_key_scores(queries, out_dev=scores.device)
            )

    def new_key_scores(self, queries, texts):
        return queries @ self.ix.key_vectors(list(texts)).to(queries.device).T

    @property
    def entry_count(self):
        return self.N + int(self.ix.key_count)


class LexicalLiveIndex(ScoringView):

    def __init__(self, retriever, counts, owner, ktexts):
        from selfindex.retrieval.bm25_tensor import TensorBM25

        self.retriever = retriever
        self.N = int(retriever.N)
        self.device = compute_device()
        self.g = TensorBM25(counts, device=self.device)
        self.owner = np.asarray(owner, np.int64)
        self.ktexts = list(ktexts)

    def query_repr(self, texts):
        lucene = self.retriever._an
        return [lucene.analyze(text) for text in texts]

    def doc_query_repr(self, docs):
        return [list(self.retriever.ix.corpus[int(d)]) for d in docs]

    def scores_rows(self, toks):
        return self.g.score_batch(toks)

    @cached_property
    def _original_scorer(self):
        from selfindex.retrieval.bm25_tensor import TermCounts, TensorBM25

        return TensorBM25(TermCounts(self.retriever.ix.corpus), device=self.device)

    def original_scores(self, queries):
        return self._original_scorer.score_batch(queries)

    def key_scores(self, toks, kids):
        import torch

        scores = self.scores_rows(toks)
        cols = [int(k) if int(k) >= 0 else -int(k) - 1 for k in kids]
        return scores[:, torch.as_tensor(cols, device=scores.device)]

    def document_scores(self, toks):
        import torch

        scores = self.scores_rows(toks)
        owners = torch.as_tensor(self.owner[self.N :], device=scores.device)
        return (
            scores[:, : self.N]
            .clone()
            .scatter_reduce_(
                1, owners.expand(len(toks), -1), scores[:, self.N :], reduce="amax"
            )
        )

    def new_key_scores(self, toks, texts):
        return self.g.score_rows(toks, self.query_repr(texts))

    @property
    def entry_count(self):
        return len(self.owner)

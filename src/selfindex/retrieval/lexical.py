from __future__ import annotations

import numpy as np

from selfindex.index.contracts import SnapshotLock
from selfindex.retrieval.bm25 import BM25Index, analyzer
from selfindex.retrieval.contracts import Retriever
from selfindex.runtime.platform.devices import compute_device

KEY_ENTRIES = ("all", "joined")


class LexicalRetriever(Retriever):

    def __init__(self, texts: list[str], entries: str = "all"):
        if entries not in KEY_ENTRIES:
            raise ValueError(
                f"unknown key entries: {entries}; available: {', '.join(KEY_ENTRIES)}"
            )
        self.entries = entries
        self._lock = SnapshotLock()
        self.N = len(texts)
        self._an = analyzer()
        self.ix = BM25Index(texts)
        self.doc_keys: dict[int, list[str]] = {}
        self._eval_ix = None
        self._live_ix = None
        self._joined_key_tokens: dict[int, tuple] = {}
        self._key_tokens: dict[str, list] = {}
        self._text_counts = None

    def snapshot(self, texts):
        from selfindex.retrieval.snapshots import LexicalSnapshot

        return LexicalSnapshot(self, texts)

    def live(self):
        with self._lock:
            return self._live_view()

    def _live_view(self):
        from selfindex.retrieval.bm25_tensor import TermCounts
        from selfindex.retrieval.views import LexicalLiveIndex

        if self._live_ix is not None:
            return self._live_ix
        if self._text_counts is None:
            self._text_counts = TermCounts(self.ix.corpus)
        owner = list(range(self.N))
        key_rows, key_texts = [], []
        for doc, keys in self.doc_keys.items():
            for key in keys:
                key_rows.append(self._key_row(key))
                owner.append(int(doc))
                key_texts.append(key)
        counts = TermCounts(key_rows, base=self._text_counts)
        self._live_ix = LexicalLiveIndex(self, counts, owner, key_texts)
        return self._live_ix

    def add_keys(self, doc: int, keys: list[str]):
        with self._lock.mutation():
            self.doc_keys.setdefault(int(doc), []).extend(
                key for key in keys if key and key.strip()
            )
            self._eval_ix = None
            self._live_ix = None
            self._joined_key_tokens.pop(int(doc), None)

    def _key_row(self, key):
        if (tokens := self._key_tokens.get(key)) is None:
            tokens = self._key_tokens[key] = self._an.analyze(key)
        return tokens

    def _joined_row(self, doc):
        keys = self.doc_keys[doc]
        cached = self._joined_key_tokens.get(doc)
        if cached is None or cached[0] != len(keys):
            cached = (len(keys), self._an.analyze("\n".join(keys)))
            self._joined_key_tokens[doc] = cached
        return cached[1]

    def _eval_index(self):
        if self._eval_ix is None:
            docs = sorted(doc for doc, keys in self.doc_keys.items() if keys)
            rows, owners = [], []
            if self.entries == "all":
                for doc in docs:
                    rows.extend(self._key_row(key) for key in self.doc_keys[doc])
                    owners.extend([doc] * len(self.doc_keys[doc]))
            rows.extend(self._joined_row(doc) for doc in docs)
            owners.extend(docs)
            self._eval_ix = (
                BM25Index(corpus=list(self.ix.corpus) + rows),
                np.asarray(owners, dtype=np.int64),
            )
        return self._eval_ix

    def score_text(self, q_text: str) -> np.ndarray:
        return self.score_queries([q_text])[0]

    def score_queries(self, texts, query_embeddings=None):
        import torch

        queries = [self._an.analyze(text) for text in texts]
        if not self.doc_keys:
            return np.stack([self.ix.score_tokens(query) for query in queries])
        index, owners = self._eval_index()
        rows = np.stack([index.score_tokens(query) for query in queries])
        device = compute_device()
        scores = torch.from_numpy(rows).to(device)
        best = scores[:, : self.N].contiguous()
        owner = torch.from_numpy(owners).to(device).expand(len(queries), -1)
        best.scatter_reduce_(1, owner, scores[:, self.N :], reduce="amax")
        return best.cpu().numpy()

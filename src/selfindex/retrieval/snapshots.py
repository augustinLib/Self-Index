from __future__ import annotations

import threading
from abc import abstractmethod
from typing import NamedTuple

import numpy as np

from selfindex.index.contracts import LiveIndexView
from selfindex.retrieval.views import LiveIndex

SNAPSHOT_QUERY_BATCH = 32
SNAPSHOT_KEY_CHUNK = 8192


class KeySearchResult(NamedTuple):
    key_ids: np.ndarray
    scores: np.ndarray


class IndexSnapshot(LiveIndexView):

    def __init__(self, retriever, texts, lock):
        self.retriever = retriever
        self.texts = tuple(texts)
        self.N = len(self.texts)
        self._lock = lock
        self._thread = None
        self._view = None
        self._entry = threading.Lock()

    @abstractmethod
    def _build_view(self): ...

    @abstractmethod
    def _key_records(self): ...

    def __enter__(self):
        if not self._entry.acquire(blocking=False):
            raise RuntimeError("snapshot lease is already active")
        self._lease = self._lock.readonly()
        self._lease.__enter__()
        try:
            self._view = self._build_view()
            if self._view.N != self.N:
                raise ValueError("snapshot corpus and index sizes differ")
            generated, self._texts, self._owners = {}, {}, {}
            for doc, kid, text in self._key_records():
                generated.setdefault(doc, []).append(kid)
                self._texts[kid], self._owners[kid] = text, doc
            self._generated = {doc: tuple(keys) for doc, keys in generated.items()}
            self._ids = np.concatenate(
                (
                    -np.arange(1, self.N + 1, dtype=np.int64),
                    np.asarray(
                        [kid for doc in sorted(generated) for kid in generated[doc]],
                        dtype=np.int64,
                    ),
                )
            )
            self._thread = threading.get_ident()
            return self
        except BaseException:
            self._view = None
            self._lease.__exit__(None, None, None)
            self._entry.release()
            raise

    def __exit__(self, *error):
        self._active()
        self._thread = None
        self._view = None
        self._lease.__exit__(*error)
        self._entry.release()

    def _active(self):
        if self._thread is None or self._view is None:
            raise RuntimeError("snapshot requires an active lease")
        return self._view

    @property
    def device(self):
        return self._active().device

    def query_repr(self, texts):
        return self._active().query_repr(texts)

    def doc_query_repr(self, docs):
        return self._active().doc_query_repr(docs)

    def original_scores(self, queries):
        return self._active().original_scores(queries)

    def key_scores(self, queries, kids):
        return self._active().key_scores(queries, kids)

    def document_scores(self, queries):
        return self._active().document_scores(queries)

    def new_key_scores(self, queries, texts):
        return self._active().new_key_scores(queries, texts)

    @property
    def entry_count(self):
        return self._active().entry_count

    def key_text(self, kid):
        self._active()
        return self.texts[self.key_owner(kid)] if kid < 0 else self._texts[kid]

    def key_owner(self, kid):
        self._active()
        if kid < 0:
            owner = -int(kid) - 1
            if not 0 <= owner < self.N:
                raise KeyError(kid)
            return owner
        return self._owners[kid]

    def keys_of(self, doc):
        self._active()
        return self._generated.get(doc, ())

    def _score_chunks(self, queries):
        for start in range(0, len(self._ids), SNAPSHOT_KEY_CHUNK):
            kids = self._ids[start : start + SNAPSHOT_KEY_CHUNK]
            yield start, self.key_scores(queries, kids)

    def retrieve_keys(self, texts, depth):
        self._active()
        if texts and not self.N:
            raise ValueError(
                "key retrieval requires positive sizes and a nonempty index"
            )
        depth = min(depth, len(self._ids))
        ids = np.empty((len(texts), depth), dtype=np.int64)
        scores = np.empty((len(texts), depth), dtype=np.float32)
        all_ids = np.asarray(self._ids, dtype=np.int64)
        for start in range(0, len(texts), SNAPSHOT_QUERY_BATCH):
            queries = self.query_repr(texts[start : start + SNAPSHOT_QUERY_BATCH])
            positions, values = self._top_keys(queries, depth)
            end = start + len(values)
            ids[start:end], scores[start:end] = all_ids[positions], values
        ids.setflags(write=False)
        scores.setflags(write=False)
        return KeySearchResult(ids, scores)

    def _top_keys(self, queries, depth):
        import torch

        positions = values = None
        for start, block in self._score_chunks(queries):
            block = torch.as_tensor(block).detach().float()
            if not torch.isfinite(block).all():
                raise ValueError("key retrieval received non-finite scores")
            columns = torch.arange(
                start, start + block.shape[1], device=block.device
            ).expand_as(block)
            if values is not None:
                block = torch.cat((values, block), dim=1)
                columns = torch.cat((positions, columns), dim=1)
            order = torch.sort(-block, dim=1, stable=True).indices[:, :depth]
            values = torch.gather(block, 1, order)
            positions = torch.gather(columns, 1, order)
        return positions.cpu().numpy(), values.cpu().numpy()


class DenseSnapshot(IndexSnapshot):
    def __init__(self, retriever, texts):
        super().__init__(retriever, texts, retriever.ix._lock)

    def _build_view(self):
        return LiveIndex(self.retriever, self.texts)

    def _key_records(self):
        return self.retriever.ix.iter_key_records()


class LexicalSnapshot(IndexSnapshot):
    def __init__(self, retriever, texts):
        super().__init__(retriever, texts, retriever._lock)

    def _build_view(self):
        return self.retriever.live()

    def _key_records(self):
        for offset, text in enumerate(self._view.ktexts):
            kid = self.N + offset
            yield int(self._view.owner[kid]), kid, text

    def _score_chunks(self, queries):
        scores = self._active().scores_rows(queries)
        for start in range(0, len(self._ids), SNAPSHOT_KEY_CHUNK):
            kids = self._ids[start : start + SNAPSHOT_KEY_CHUNK]
            columns = [kid if kid >= 0 else -kid - 1 for kid in kids]
            yield start, scores[:, columns]

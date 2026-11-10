from __future__ import annotations

import json
import threading

import numpy as np

import longmem.adapter._bootstrap  # noqa: F401
from longmem.adapter.key_state import MemoryKeyState, entry_keys, load_keys
from longmem.adapter.retrieval import MaxFusionRetriever

from memory_modules.memory import register_memory
from memory_modules.rag import RagMemory
from memory_modules.agentrunbook_r import AgentRunbookR


class SelfIndexMixin:

    def __init__(self, memory_params):
        params = dict(memory_params)
        keys_path = params.pop("selfindex_keys_path", None)
        super().__init__(params)
        self._selfindex_init(keys_path)

    def _save_backend(self, output_dir):
        with self._si_lock:
            super()._save_backend(output_dir)
            MemoryKeyState(self._si_key_emb, self._si_key_owner, self._si_keys).save(
                output_dir
            )

    def _load_backend(self, input_dir):
        with self._si_lock:
            super()._load_backend(input_dir)
            state = MemoryKeyState.load(
                input_dir, self.raw_state_entries, self.raw_state_embeddings
            )
            requested = json.loads(json.dumps(self._si_keys))
            if requested and requested != state.by_entry:
                raise ValueError(
                    "requested keys differ from the saved memory; "
                    "rebuild instead of mixing key sets"
                )
            self._si_key_emb = state.embeddings
            self._si_key_owner = state.owners
            self._si_keys = state.by_entry

    def _selfindex_init(self, keys_path):
        self._si_lock = threading.RLock()
        self._si_keys: dict[str, list[str]] = load_keys(keys_path) if keys_path else {}
        self._si_key_owner: list[int] = []
        self._si_key_emb = np.zeros((0, 0), np.float32)

    def insert(self, trajectory):
        with self._si_lock:
            n_before = len(self.raw_state_entries)
            super().insert(trajectory)
            texts, owners = entry_keys(self.raw_state_entries, self._si_keys, n_before)
            if not texts:
                return
            key_embeddings = np.asarray(
                self._embed_texts(texts, is_query=False), np.float32
            )
            self._si_key_emb = (
                key_embeddings
                if self._si_key_emb.size == 0
                else np.vstack([self._si_key_emb, key_embeddings])
            )
            self._si_key_owner.extend(owners)

    def _search_entries(self, *, entries, embeddings, query_text, top_k):
        ranked = self._rank_raw_state_pool(entries, embeddings, query_text, top_k)
        if ranked is None:
            return super()._search_entries(
                entries=entries,
                embeddings=embeddings,
                query_text=query_text,
                top_k=top_k,
            )
        return [
            {"query": query_text, "rank": rank, "score": score, "entry": entries[index]}
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    def _rank_raw_state_pool(self, entries, embeddings, query_text, top_k):
        base_pool = embeddings is getattr(self, "raw_state_embeddings", None)
        if (
            not base_pool
            or self._si_key_emb.size == 0
            or not len(entries)
            or not query_text.strip()
        ):
            return None
        query_vector = self._embed_texts([query_text], is_query=True)
        if query_vector.size == 0:
            return []
        query = np.asarray(query_vector[0], np.float32)
        return MaxFusionRetriever().rank(
            query, embeddings, self._si_key_emb, np.asarray(self._si_key_owner), top_k
        )


@register_memory
class RagSelfIndex(SelfIndexMixin, RagMemory):
    memory_type = "rag_selfindex"


@register_memory
class ArrSelfIndex(SelfIndexMixin, AgentRunbookR):
    memory_type = "agentrunbook_r_selfindex"

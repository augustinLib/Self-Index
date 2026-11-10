from __future__ import annotations

from selfindex.index.contracts import KeyIndex


class LexicalIndexAdapter(KeyIndex):
    has_embeddings = False

    def __init__(self, retriever):
        self.ret = retriever
        self._lock = retriever._lock
        self.N = retriever.N
        self.accepted_edits = [0] * self.N
        self.generated_keys: dict[int, list[str]] = {}

    def add_keys(self, doc: int, texts: list[str]):
        keys = [key for key in (texts or []) if isinstance(key, str) and key.strip()]
        if not keys:
            return
        with self._lock.mutation():
            self.ret.add_keys(int(doc), keys)
            self.generated_keys.setdefault(int(doc), []).extend(keys)
            self.accepted_edits[int(doc)] += 1

    def replace_validated(self, doc, texts):
        doc, keys = self._validated_replacement(doc, texts)
        with self._lock.mutation():
            self.ret.doc_keys[doc] = list(keys)
            self.generated_keys[doc] = keys
            self.ret._joined_key_tokens.pop(doc, None)
            self.ret._eval_ix = self.ret._live_ix = None
            self.accepted_edits[doc] += 1

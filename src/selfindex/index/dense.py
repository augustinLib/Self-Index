from __future__ import annotations

import numpy as np

from selfindex.index.contracts import KeyIndex, SnapshotLock
from selfindex.runtime.platform.devices import compute_device, devices
from selfindex.settings import KEY_MAX_CHARS


class IndexState(KeyIndex):
    has_embeddings = True

    def __init__(self, doc_texts: list[str], encode_fn):
        self.N = len(doc_texts)
        self._encode = encode_fn
        self.emb_orig = encode_fn(doc_texts, is_query=False)
        dim = self.emb_orig.shape[1]
        self._initialize_tensor_mirror()
        self._initialize_key_shards(dim)
        self.accepted_edits = [0] * self.N
        self._lock = SnapshotLock()

    def _initialize_key_shards(self, dim):
        t = self._torch
        self._key_dim = dim
        self._key_devices = self._shard_devices()
        self._key_vectors = [
            t.zeros((0, dim), dtype=t.float32, device=d) for d in self._key_devices
        ]
        self._key_owners = [
            t.zeros((0,), dtype=t.long, device=d) for d in self._key_devices
        ]
        self._key_alive = [
            t.zeros((0,), dtype=t.bool, device=d) for d in self._key_devices
        ]
        self._key_counts = [0] * len(self._key_devices)
        self._key_texts: list[list] = [[] for _ in self._key_devices]
        self._key_capacity = [0] * len(self._key_devices)
        if len(self._key_devices) > 1:
            print(
                f"[index] key storage sharded over {len(self._key_devices)} GPUs: "
                f"{', '.join(self._key_devices)}",
                flush=True,
            )
        self.generated_keys: dict[int, list[str]] = {}
        self._encoded_keys = {}

    def _shard_devices(self):
        return devices()

    def _initialize_tensor_mirror(self):
        import torch

        self._torch = torch
        self._dev = compute_device()
        self.emb_orig_t = torch.from_numpy(np.ascontiguousarray(self.emb_orig)).to(
            self._dev
        )

    def _clip_keys(self, texts):
        return [
            text[:KEY_MAX_CHARS]
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

    @property
    def key_count(self):
        return int(sum(self._key_counts))

    def _shard_scores(self, queries, shard):
        t = self._torch
        device, count = self._key_devices[shard], self._key_counts[shard]
        batch = int(queries.shape[0])
        on_device = (
            queries
            if str(queries.device) == device
            else queries.to(device, non_blocking=True)
        )
        scores = on_device @ self._key_vectors[shard][:count].T
        dead = ~self._key_alive[shard][:count]
        if bool(dead.any()):
            scores[:, dead] = -1e9
        owners = self._key_owners[shard][:count].unsqueeze(0).expand(batch, -1)
        document_max = t.full((batch, self.N), -1e9, device=device)
        document_max.scatter_reduce_(1, owners, scores, reduce="amax")
        return scores, owners, document_max

    def best_key_scores(self, q, out_dev):
        t = self._torch
        target = out_dev
        if q.dim() == 1:
            q = q.unsqueeze(0)
        result = t.full((int(q.shape[0]), self.N), -1e9, device=target)
        if self.key_count == 0:
            return result
        parts = []
        for shard, device in enumerate(self._key_devices):
            if not self._key_counts[shard]:
                continue
            parts.append((device, self._shard_scores(q, shard)[2]))
        for device, part in parts:
            t.maximum(
                result, part if device == str(target) else part.to(target), out=result
            )
        del parts
        return result

    KID_STRIDE = 1 << 40
    MIN_BUFFER_ROWS = 64
    MAX_GROWTH_BYTES = 64 << 20

    def _live_rows_of(self, shard, count, doc):
        owned = self._key_owners[shard][:count] == int(doc)
        return owned & self._key_alive[shard][:count]

    def keys_of(self, doc):
        out = []
        for shard, count in enumerate(self._key_counts):
            if not count:
                continue
            rows = self._live_rows_of(shard, count, doc).nonzero().flatten().tolist()
            out.extend(shard * self.KID_STRIDE + row for row in rows)
        return out

    def iter_key_records(self):
        with self._lock:
            for shard, count in enumerate(self._key_counts):
                rows = self._key_alive[shard][:count].nonzero().flatten()
                owners = self._key_owners[shard][rows].cpu().tolist()
                for row, owner in zip(rows.cpu().tolist(), owners, strict=False):
                    yield (
                        owner,
                        shard * self.KID_STRIDE + row,
                        self._key_texts[shard][row],
                    )

    def kid_rows(self, kids, dev=None):
        t = self._torch
        dev = dev or self._dev
        kids = [int(k) for k in kids]
        if not kids:
            return t.zeros((0, self._key_dim), device=dev)
        out = t.empty((len(kids), self._key_dim), dtype=t.float32, device=dev)
        by_shard, text_positions, text_docs = self._group_kids(kids)
        for shard, pairs in by_shard.items():
            positions = t.as_tensor([position for position, _ in pairs], device=dev)
            rows = t.as_tensor(
                [row for _, row in pairs], device=self._key_devices[shard]
            )
            out[positions] = self._key_vectors[shard][rows].to(dev)
        if text_positions:
            positions = t.as_tensor(text_positions, device=dev)
            out[positions] = self._original_rows(text_docs, dev)
        return out

    def _group_kids(self, kids):
        by_shard: dict[int, list] = {}
        text_positions, text_docs = [], []
        for position, kid in enumerate(kids):
            if kid < 0:
                text_positions.append(position)
                text_docs.append(-kid - 1)
            else:
                shard, row = divmod(kid, self.KID_STRIDE)
                by_shard.setdefault(shard, []).append((position, row))
        return by_shard, text_positions, text_docs

    def _original_rows(self, docs, device):
        mirror = self.emb_orig_t
        return mirror[self._torch.as_tensor(docs, device=mirror.device)].to(device)

    def _append_keys(self, vectors, owners, texts=None):
        k = int(vectors.shape[0])
        shard = min(range(len(self._key_devices)), key=lambda j: self._key_counts[j])
        device = self._key_devices[shard]
        if str(vectors.device) != device:
            vectors = vectors.to(device)
            owners = owners.to(device)
        count = self._key_counts[shard]
        if count + k > self._key_capacity[shard]:
            self._grow_shard(shard, count + k)
        self._key_vectors[shard][count : count + k] = vectors
        self._key_owners[shard][count : count + k] = owners
        self._key_alive[shard][count : count + k] = True
        self._key_counts[shard] = count + k
        row_texts = list(texts) if texts is not None and len(texts) == k else [None] * k
        self._key_texts[shard].extend(row_texts)

    def _grow_shard(self, shard, needed):
        t = self._torch
        device = self._key_devices[shard]
        row_bytes = self._key_dim * 4 + 8 + 1
        slack = min(needed // 2, self.MAX_GROWTH_BYTES // row_bytes)
        capacity = max(self.MIN_BUFFER_ROWS, needed + slack)
        vectors = t.empty((capacity, self._key_dim), dtype=t.float32, device=device)
        owners = t.empty((capacity,), dtype=t.long, device=device)
        alive = t.zeros((capacity,), dtype=t.bool, device=device)
        count = self._key_counts[shard]
        vectors[:count].copy_(self._key_vectors[shard][:count])
        owners[:count].copy_(self._key_owners[shard][:count])
        alive[:count].copy_(self._key_alive[shard][:count])
        self._key_vectors[shard], self._key_owners[shard] = vectors, owners
        self._key_alive[shard], self._key_capacity[shard] = alive, capacity

    def add_keys(self, doc: int, texts: list[str]):
        keys = self._clip_keys(texts)
        if not keys:
            return
        self._store_keys(doc, keys, self._encode(keys, is_query=False))

    def _store_keys(self, doc, texts, embeddings):
        t = self._torch
        vectors = t.from_numpy(np.ascontiguousarray(embeddings.astype(np.float32))).to(
            self._dev
        )
        owners = t.full((len(texts),), doc, dtype=t.long, device=self._dev)
        with self._lock.mutation():
            self._append_keys(vectors, owners, texts)
            self.accepted_edits[doc] += 1
            self.generated_keys.setdefault(doc, []).extend(texts)

    def retire_kids(self, kids):
        with self._lock.mutation():
            for kid in kids:
                kid = int(kid)
                if kid < 0:
                    continue
                shard, row = divmod(kid, self.KID_STRIDE)
                if row >= self._key_counts[shard] or not bool(
                    self._key_alive[shard][row]
                ):
                    continue
                self._key_alive[shard][row] = False
                owner = int(self._key_owners[shard][row].item())
                shard_texts = self._key_texts[shard]
                text = shard_texts[row] if row < len(shard_texts) else None
                owner_texts = self.generated_keys.get(owner)
                if owner_texts and text in owner_texts:
                    owner_texts.remove(text)

    def replace_validated_many(self, replacements):
        t = self._torch
        checked = [
            self._validated_replacement(doc, keys) for doc, keys in replacements.items()
        ]
        with self._lock:
            held = {doc: self._held_keys(doc) for doc, _ in checked}
            kept = list(
                dict.fromkeys(
                    held[doc][key]
                    for doc, keys in checked
                    for key in keys
                    if key in held[doc]
                )
            )
            kept_rows = self.kid_rows(kept)
        fresh = list(
            dict.fromkeys(
                key for doc, keys in checked for key in keys if key not in held[doc]
            )
        )
        try:
            vectors = t.cat([kept_rows, self.key_vectors(fresh)])
        finally:
            self._encoded_keys.clear()
        position = {kid: row for row, kid in enumerate(kept)}
        position.update({key: len(kept) + row for row, key in enumerate(fresh)})
        for doc, keys in checked:
            rows = [
                position[held[doc][key]] if key in held[doc] else position[key]
                for key in keys
            ]
            index = t.as_tensor(rows, dtype=t.long, device=vectors.device)
            self._replace_encoded(doc, keys, vectors[index])

    def _held_keys(self, doc):
        held = {}
        for kid in self.keys_of(doc):
            shard, row = divmod(kid, self.KID_STRIDE)
            held.setdefault(self._key_texts[shard][row], kid)
        return held

    def key_vectors(self, keys):
        t = self._torch
        fresh = [key for key in dict.fromkeys(keys) if key not in self._encoded_keys]
        if fresh:
            embeddings = np.asarray(
                self._encode(fresh, is_query=False), dtype=np.float32
            )
            if (
                embeddings.shape != (len(fresh), self._key_dim)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("key embeddings have invalid shape or values")
            vectors = t.from_numpy(np.ascontiguousarray(embeddings)).to(self._dev)
            self._encoded_keys.update(zip(fresh, vectors))
        if not keys:
            return t.zeros((0, self._key_dim), dtype=t.float32, device=self._dev)
        return t.stack([self._encoded_keys[key] for key in keys])

    def _replace_encoded(self, doc, keys, vectors):
        owners = self._torch.full(
            (len(keys),), doc, dtype=self._torch.long, device=self._dev
        )
        with self._lock.mutation():
            previous = self.keys_of(doc)
            counts = list(self._key_counts)
            try:
                if keys:
                    self._append_keys(vectors, owners, keys)
            except BaseException:
                for shard, count in enumerate(counts):
                    self._key_counts[shard] = count
                    del self._key_texts[shard][count:]
                raise
            self.retire_kids(previous)
            self.generated_keys[doc] = keys
            self.accepted_edits[doc] += 1

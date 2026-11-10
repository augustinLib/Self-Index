import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from selfindex.common.io import read_lines
from selfindex.storage.artifacts import atomic_replace

STATE_FILENAME = "selfindex_keys.npz"
STATE_VERSION = 3


def load_keys(path):
    by_entry = {}
    for number, line in enumerate(read_lines(path), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        key, entry = record.get("k"), record.get("docid")
        if not (
            isinstance(key, str) and key.strip() and isinstance(entry, str) and entry
        ):
            raise ValueError(f"invalid key record at {path}:{number}")
        by_entry.setdefault(entry, []).append(key)
    return by_entry


def entry_keys(entries, by_entry, start=0):
    texts, owners = [], []
    for row in range(start, len(entries)):
        for key in by_entry.get(entries[row]["entry_id"], ()):
            texts.append(key)
            owners.append(row)
    return texts, owners


def _check_keys(embeddings, owners, entries, raw_embeddings):
    if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("invalid SelfIndex key owners")
    dimension = raw_embeddings.shape[1] if owners.size else None
    valid = (
        embeddings.ndim == 2
        and embeddings.shape[0] == len(owners)
        and (dimension is None or embeddings.shape[1] == dimension)
        and np.issubdtype(embeddings.dtype, np.floating)
        and np.isfinite(embeddings).all()
    )
    if not valid or np.any((owners < 0) | (owners >= len(entries))):
        raise ValueError("invalid SelfIndex key embeddings or owner bounds")


@dataclass
class MemoryKeyState:
    embeddings: np.ndarray
    owners: list[int]
    by_entry: dict

    def save(self, directory):
        path = Path(directory) / STATE_FILENAME
        record = {"version": STATE_VERSION, "by_entry": self.by_entry}
        metadata = json.dumps(record, ensure_ascii=True)
        atomic_replace(
            path,
            lambda stream: np.savez_compressed(
                stream,
                embeddings=self.embeddings,
                owners=np.asarray(self.owners, dtype=np.int64),
                metadata=metadata,
            ),
            suffix=".npz",
        )

    @classmethod
    def load(cls, directory, entries, raw_embeddings):
        path = Path(directory) / STATE_FILENAME
        if not path.is_file():
            raise ValueError(
                "memory checkpoint has no SelfIndex key state; "
                "rebuild the memory with its key file"
            )
        with np.load(path, allow_pickle=False) as archive:
            embeddings, owners = archive["embeddings"], archive["owners"]
            metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("version") != STATE_VERSION:
            raise ValueError("SelfIndex key state does not match its format")
        _check_keys(embeddings, owners, entries, raw_embeddings)
        by_entry = metadata.get("by_entry")
        if not isinstance(by_entry, dict):
            raise ValueError("invalid SelfIndex entry mapping")
        return cls(embeddings, owners.tolist(), by_entry)

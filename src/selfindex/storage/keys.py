from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

from selfindex.storage.artifacts import LocalArtifactStore


def write_keys(path, keys_by_document):
    if not keys_by_document:
        return
    with Path(path).open("w") as stream:
        for document, keys in keys_by_document.items():
            record = {"doc": int(document), "keys": list(keys)}
            record_line = json.dumps(record) + "\n"
            stream.write(record_line)


def key_records(keys_by_document, document_ids):
    for document, keys in keys_by_document.items():
        if not 0 <= document < len(document_ids):
            raise ValueError(f"invalid key owner: {document}")
        for key in keys:
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"invalid live key for document {document}")
            yield {"k": key, "docid": str(document_ids[document])}


def publish_keys(keys_by_document, document_ids, destination):
    records = iter(key_records(keys_by_document, document_ids))
    if (first := next(records, None)) is None:
        raise ValueError("optimization produced no live keys to publish")
    path = Path(destination)
    LocalArtifactStore(path.parent).write_jsonl(path.name, chain([first], records))

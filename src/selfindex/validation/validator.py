from collections import Counter

import numpy as np

from selfindex.validation.base import EncodedTexts, KeyChunk
from selfindex.validation.separation import SeparationCriterion
from selfindex.validation.specificity import SpecificityCriterion

SCORE_MEMORY_BYTES = 2 << 30


SCORE_ELEMENT_BYTES = np.dtype(np.float32).itemsize


MIN_CHUNK_KEYS = 64


MAX_CHUNK_KEYS = 8192


def _chunk_size(n_docs):
    capacity = SCORE_MEMORY_BYTES // (SCORE_ELEMENT_BYTES * max(n_docs, 1))
    return max(MIN_CHUNK_KEYS, min(MAX_CHUNK_KEYS, capacity))


def validate_scores(live, proposals, previous_keys, competitors, *, depth):
    stats = Counter()
    docs = [int(doc) for doc, keys in proposals.items() if keys]
    if not docs:
        return {}, stats
    owners, texts = zip(*((doc, text) for doc in docs for text in proposals[doc]))
    previous = {
        doc: tuple(
            text
            for text in (live.key_text(key) for key in previous_keys.get(doc) or ())
            if text
        )
        for doc in docs
        if competitors.get(doc)
    }
    encoded = EncodedTexts(
        live, [*texts, *(text for keys in previous.values() for text in keys)]
    )
    criteria = (
        SpecificityCriterion(live, depth),
        SeparationCriterion(live, previous, competitors, encoded),
    )
    accepted = _apply_criteria(live, criteria, encoded, texts, owners, stats)
    kept = {}
    for doc, text, keep in zip(owners, texts, accepted):
        if keep:
            kept.setdefault(doc, []).append(text)
    stats["keys_checked"] = len(texts)
    stats["keys_kept"] = sum(map(len, kept.values()))
    return kept, stats


def _apply_chunk(live, criteria, chunk, stats):
    import torch

    keep = torch.ones(len(chunk.texts), dtype=torch.bool, device=live.device)
    for criterion in criteria:
        criterion.apply(chunk, keep, stats)
    return keep.cpu().numpy()


def _apply_criteria(live, criteria, encoded, texts, owners, stats):
    accepted = np.ones(len(texts), dtype=bool)
    chunk_size = _chunk_size(live.entry_count)
    for start in range(0, len(texts), chunk_size):
        end = min(len(texts), start + chunk_size)
        chunk_texts = texts[start:end]
        chunk = KeyChunk(chunk_texts, owners[start:end], encoded(chunk_texts))
        accepted[start:end] = _apply_chunk(live, criteria, chunk, stats)
    return accepted

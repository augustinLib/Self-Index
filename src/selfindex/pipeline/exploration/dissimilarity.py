from __future__ import annotations

import numpy as np

from selfindex.exploration.dissimilarity import JACCARD_TAU, Dissimilarity, words
from selfindex.runtime.platform.devices import compute_device

CANDIDATE_BLOCK_ROWS = 2048
USED_BLOCK_ROWS = 8192
FP16_EXACT_INTEGER_LIMIT = 2048
GREEDY_TRANSFER_ROWS = 256


def _dense_tokens(block, width, dtype, device):
    import torch

    dense = torch.zeros(len(block), width, dtype=dtype, device=device)
    if block:
        lengths = torch.tensor([len(b) for b in block])
        rows = torch.repeat_interleave(torch.arange(len(block)), lengths).to(device)
        cols = torch.tensor([w for b in block for w in b], device=device)
        if len(cols):
            dense[rows, cols] = 1.0
    return dense


def _greedy_accept(pair, unavailable):
    count = len(unavailable)
    accepted = np.zeros(count, dtype=bool)
    blocked = unavailable.cpu().numpy()
    for start in range(0, count, GREEDY_TRANSFER_ROWS):
        end = min(start + GREEDY_TRANSFER_ROWS, count)
        rows = pair[start:end, :end].cpu().numpy()
        for offset, row in enumerate(rows):
            candidate = start + offset
            if not blocked[candidate] and not np.any(
                row[:candidate] & accepted[:candidate]
            ):
                accepted[candidate] = True
    return accepted.tolist()


def _tensor_mask(used_texts, cand_texts, dev: str):
    import torch

    used_ids, cand_ids, width = _token_ids(used_texts, cand_texts)
    n_c = len(cand_ids)
    if n_c == 0:
        return []
    dtype = (
        torch.float32
        if max(map(len, used_ids + cand_ids), default=0) > FP16_EXACT_INTEGER_LIMIT
        else torch.float16
    )
    size_c = torch.tensor([len(s) for s in cand_ids], device=dev, dtype=torch.float32)
    size_u = torch.tensor([len(s) for s in used_ids], device=dev, dtype=torch.float32)
    rows = CANDIDATE_BLOCK_ROWS
    blocks = _JaccardBlocks(width, dtype, dev, size_c)
    cblocks = [
        (start, _dense_tokens(cand_ids[start : start + rows], width, dtype, dev))
        for start in range(0, n_c, rows)
    ]
    dup_used = blocks.used_duplicates(cblocks, used_ids, size_u)
    pair = blocks.candidate_pairs(cblocks)
    del cblocks
    out = _greedy_accept(pair, dup_used)
    del pair, dup_used
    return out


def _token_ids(used_texts, cand_texts):
    used_sets = [words(t) for t in used_texts]
    cand_sets = [words(t) for t in cand_texts]
    vocab: dict[str, int] = {}

    def ids(word_set):
        return [vocab.setdefault(w, len(vocab)) for w in word_set]

    used_ids = [ids(s) for s in used_sets if s]
    cand_ids = [ids(s) for s in cand_sets]
    return used_ids, cand_ids, max(len(vocab), 1)


class _JaccardBlocks:

    def __init__(self, width, dtype, device, cand_sizes):
        self.width = width
        self.dtype = dtype
        self.device = device
        self.cand_sizes = cand_sizes

    def hits(self, inter, size_a, size_b):
        i = inter.double()
        union = (size_a[:, None] + size_b[None, :]).double() - i
        return (
            (i / union.clamp(min=1.0) >= JACCARD_TAU)
            & (size_a[:, None] > 0)
            & (size_b[None, :] > 0)
        )

    def used_duplicates(self, cblocks, used_ids, used_sizes):
        import torch

        n_c = len(self.cand_sizes)
        rows = CANDIDATE_BLOCK_ROWS
        dup_used = torch.zeros(n_c, dtype=torch.bool, device=self.device)
        for start in range(0, len(used_ids), USED_BLOCK_ROWS):
            used = _dense_tokens(
                used_ids[start : start + USED_BLOCK_ROWS],
                self.width,
                self.dtype,
                self.device,
            )
            sizes = used_sizes[start : start + USED_BLOCK_ROWS]
            for cand_start, cand_tokens in cblocks:
                inter = cand_tokens @ used.T
                dup_used[cand_start : cand_start + rows] |= self.hits(
                    inter, self.cand_sizes[cand_start : cand_start + rows], sizes
                ).any(dim=1)
                del inter
            del used
        return dup_used

    def candidate_pairs(self, cblocks):
        import torch

        n_c = len(self.cand_sizes)
        rows = CANDIDATE_BLOCK_ROWS
        pair = torch.zeros(n_c, n_c, dtype=torch.bool, device=self.device)
        for start_a, cand_a in cblocks:
            for start_b, cand_b in cblocks:
                inter = cand_a @ cand_b.T
                pair[start_a : start_a + rows, start_b : start_b + rows] = self.hits(
                    inter,
                    self.cand_sizes[start_a : start_a + rows],
                    self.cand_sizes[start_b : start_b + rows],
                )
                del inter
        return pair


def dissimilar_mask(used_texts, cand_texts):
    selected = compute_device()
    if str(selected).startswith("cuda"):
        return _tensor_mask(used_texts, cand_texts, selected)
    return _scan_mask(used_texts, cand_texts)


def _scan_mask(used_texts, cand_texts):
    dissimilarity = Dissimilarity()
    for text in used_texts:
        dissimilarity.add(text)
    out = []
    for query in cand_texts:
        if accepted := dissimilarity.is_novel(query):
            dissimilarity.add(query)
        out.append(accepted)
    return out

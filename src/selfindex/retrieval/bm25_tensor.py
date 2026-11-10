from __future__ import annotations

import math
from array import array
from collections import Counter

import numpy as np
import torch

from selfindex.runtime.platform import settings as C


class TermCounts:

    def __init__(self, token_rows, base: TermCounts | None = None):
        self.base = base
        self.first_row = base.n_rows if base is not None else 0
        self.first_term = base.n_terms if base is not None else 0
        self.vocab: dict[str, int] = {}
        rows, terms, frequencies = array("q"), array("q"), array("q")
        lengths = np.zeros(len(token_rows), np.int64)
        for offset, tokens in enumerate(token_rows):
            lengths[offset] = len(tokens)
            counts = Counter(tokens)
            rows.extend([self.first_row + offset] * len(counts))
            terms.extend([self._number(token) for token in counts])
            frequencies.extend(counts.values())
        self.rows = np.frombuffer(rows, np.int64)
        self.terms = np.frombuffer(terms, np.int64)
        self.frequencies = np.frombuffer(frequencies, np.int64)
        self.lengths = lengths
        self.n_rows = self.first_row + len(token_rows)
        self.n_terms = self.first_term + len(self.vocab)

    def term(self, token):
        term = self.base.term(token) if self.base is not None else None
        return self.vocab.get(token) if term is None else term

    def _number(self, token):
        term = self.term(token)
        if term is None:
            term = self.vocab[token] = self.first_term + len(self.vocab)
        return term

    def parts(self):
        return (*self.base.parts(), self) if self.base is not None else (self,)


class TensorBM25:

    def __init__(self, counts: TermCounts, device):
        self.counts = counts
        self.dev = device
        self.k1, self.b = float(C.BM25_K1), float(C.BM25_B)
        self.N = counts.n_rows
        parts = counts.parts()
        rows, terms, frequencies, lengths = (
            torch.cat([torch.from_numpy(getattr(part, name)) for part in parts]).to(
                self.dev
            )
            for name in ("rows", "terms", "frequencies", "lengths")
        )
        self.avgdl = int(lengths.sum()) / self.N if self.N else 0.0
        document_frequency = torch.bincount(terms, minlength=counts.n_terms)
        self.idf = math.log(self.N + 1.0) - torch.log(document_frequency.double() + 0.5)
        weights = self._weights(frequencies, lengths.double()[rows], terms)
        self.W = (
            torch.sparse_coo_tensor(
                torch.stack([terms, rows]), weights, (counts.n_terms, self.N)
            )
            .coalesce()
            .to_sparse_csr()
        )

    def _weights(self, frequencies, lengths, terms):
        frequencies = frequencies.double()
        saturation = frequencies / (
            frequencies + self.k1 * (1 - self.b + self.b * lengths / self.avgdl)
        )
        return (self.idf[terms] * saturation).float()

    def known_terms(self, batch_tokens):
        queries, terms, counts = array("q"), array("q"), array("q")
        for query, tokens in enumerate(batch_tokens):
            known = Counter(
                term for term in map(self.counts.term, tokens) if term is not None
            )
            queries.extend([query] * len(known))
            terms.extend(known.keys())
            counts.extend(known.values())
        return tuple(
            torch.from_numpy(np.frombuffer(values, np.int64)).to(self.dev)
            for values in (queries, terms, counts)
        )

    def score_batch(self, batch_tokens):
        return self._score(self.W, batch_tokens)

    def score_rows(self, batch_tokens, rows):
        entries, terms, counts = self.known_terms(rows)
        lengths = torch.tensor(
            [len(tokens) for tokens in rows], dtype=torch.float64, device=self.dev
        )
        weights = self._weights(counts, lengths[entries], terms)
        matrix = (
            torch.sparse_coo_tensor(
                torch.stack([terms, entries]), weights, (self.counts.n_terms, len(rows))
            )
            .coalesce()
            .to_sparse_csr()
        )
        return self._score(matrix, batch_tokens)

    def _score(self, matrix, batch_tokens):
        batch = len(batch_tokens)
        queries, terms, counts = self.known_terms(batch_tokens)
        if not len(terms):
            return torch.zeros(batch, matrix.shape[1], device=self.dev)
        lengths = torch.zeros(batch, dtype=torch.float64, device=self.dev)
        lengths.scatter_add_(0, queries, counts.double())
        weights = self._weights(counts, lengths[queries], terms)
        return self._accumulate(matrix, queries, terms, weights, batch)

    def _accumulate(self, matrix, query_rows, terms, query_weights, batch_size):
        columns = matrix.shape[1]
        crow = matrix.crow_indices()
        documents = matrix.col_indices()
        values = matrix.values()
        starts, ends = crow[terms], crow[terms + 1]
        postings = ends - starts
        total = int(postings.sum().item())
        out = torch.zeros(batch_size * columns, device=self.dev, dtype=values.dtype)
        if total:
            offsets = torch.repeat_interleave(starts, postings)
            within = torch.arange(total, device=self.dev) - torch.repeat_interleave(
                torch.cumsum(postings, 0) - postings, postings
            )
            positions = offsets + within
            query_of = torch.repeat_interleave(query_rows, postings)
            weight_of = torch.repeat_interleave(query_weights, postings)
            out.scatter_add_(
                0,
                query_of * columns + documents[positions],
                values[positions] * weight_of,
            )
        return out.view(batch_size, columns)

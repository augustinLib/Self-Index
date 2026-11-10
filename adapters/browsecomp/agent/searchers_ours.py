import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from browsecomp.adapters.search import SearchAdapter
from selfindex.common.io import read_pickle
from selfindex.runtime.platform.devices import compute_device, devices
from browsecomp.agent._paths import INDEXES

SNIPPET_TOKENS = 512
BASE_INDEX = os.path.join(INDEXES, "bm25")


def document_preview(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:SNIPPET_TOKENS], skip_special_tokens=True)


class BM25Searcher(SearchAdapter):

    def __init__(self, keys, tokenizer, corpus_text):
        import jnius_config

        if keys is not None and keys.index is None:
            raise ValueError("BM25 search over keys needs the key set's --index")
        if not jnius_config.vm_running:
            jnius_config.add_options("-Xmx16g")
        from pyserini.search.lucene import LuceneSearcher

        index = str(keys.index) if keys is not None else BASE_INDEX
        self.lucene = LuceneSearcher(index)
        self.use_key = keys is not None
        self.tokenizer = tokenizer
        self.corpus_text = corpus_text

    def search(self, query, k=5):
        if self.use_key:
            top = self._best_per_document(self.lucene.search(query, 200), k)
        else:
            top = [(hit.docid, hit.score) for hit in self.lucene.search(query, k)]
        out = []
        for docid, score in top:
            text = self.corpus_text[str(docid)]
            out.append(
                {
                    "docid": str(docid),
                    "score": float(score),
                    "text": document_preview(self.tokenizer, text),
                }
            )
        return out

    @staticmethod
    def _best_per_document(hits, k):
        best = {}
        for hit in hits:
            parent = hit.docid.split("#k")[0]
            if hit.score > best.get(parent, -1e9):
                best[parent] = hit.score
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    def search_description(self, k):
        return (
            f"Retrieves the top {k} most relevant documents from the local knowledge "
            f"base using BM25 search. Provide a natural language or keyword query."
        )


class DenseSearcher(SearchAdapter):

    SCORE_FLOOR = -1e9

    def __init__(self, keys, tokenizer, corpus_text):
        import torch

        torch.backends.cuda.enable_cudnn_sdp(False)
        from browsecomp.adapters.embedding import QwenEmbBackend

        self.torch = torch
        self.use_key = keys is not None
        self.tokenizer = tokenizer
        self.corpus_text = corpus_text
        self._load_corpus()
        self.backend = QwenEmbBackend()
        if keys is not None:
            self._load_keys(keys.directory)
        self._gpu = ThreadPoolExecutor(max_workers=1)

    def _load_corpus(self):
        reps, lookup = [], []
        for path in sorted(
            glob.glob(os.path.join(INDEXES, "qwen3-embedding-8b", "corpus.shard*.pkl"))
        ):
            vectors, ids = read_pickle(path)
            reps.append(np.asarray(vectors, np.float16))
            lookup.extend(ids)
        self.document_vectors = (
            self.torch.from_numpy(np.concatenate(reps)).to(compute_device()).half()
        )
        self.lookup = [str(x) for x in lookup]

    def _load_keys(self, directory):
        with np.load(os.path.join(directory, "key_emb.npz")) as keys:
            vectors, owners = keys["v"], keys["owner"]
        self.key_devs = devices()
        partitions = self._key_partitions(len(vectors))
        self.key_vectors = self._distribute_rows(vectors, partitions)
        self.key_owners = self._distribute_rows(owners, partitions)

    def _key_partitions(self, rows):
        count = len(self.key_devs)
        bounds = [rows * index // count for index in range(count + 1)]
        return list(zip(bounds[:-1], bounds[1:]))

    def _distribute_rows(self, values, partitions):
        return [
            self.torch.from_numpy(np.ascontiguousarray(values[start:end])).to(device)
            for device, (start, end) in zip(self.key_devs, partitions, strict=False)
        ]

    def _search_gpu(self, query, k):
        torch = self.torch
        query_vector = (
            torch.from_numpy(self.backend.encode_queries([query]))
            .to(self.document_vectors.device)
            .half()
        )
        scores = (query_vector @ self.document_vectors.T).float()
        if self.use_key:
            for device, key_vectors, key_owners in zip(
                self.key_devs, self.key_vectors, self.key_owners, strict=False
            ):
                device_query = (
                    query_vector
                    if str(scores.device) == device
                    else query_vector.to(device)
                )
                key_scores = (device_query @ key_vectors.T).float()
                document_scores = torch.full(
                    (1, scores.shape[1]),
                    self.SCORE_FLOOR,
                    device=device,
                    dtype=torch.float32,
                )
                document_scores.scatter_reduce_(
                    1, key_owners.unsqueeze(0), key_scores, reduce="amax"
                )
                scores = torch.maximum(scores, document_scores.to(scores.device))
                del key_scores, document_scores
        return scores[0].topk(k)

    def search(self, query, k=5):
        scores, ix = self._gpu.submit(self._search_gpu, query, k).result()
        out = []
        for score, i in zip(scores.tolist(), ix.tolist(), strict=False):
            docid = self.lookup[i]
            out.append(
                {
                    "docid": docid,
                    "score": float(score),
                    "text": document_preview(self.tokenizer, self.corpus_text[docid]),
                }
            )
        return out

    def search_description(self, k):
        return (
            f"Retrieves the top {k} most relevant documents from the local knowledge "
            f"base using dense embedding search. Provide a natural language query."
        )

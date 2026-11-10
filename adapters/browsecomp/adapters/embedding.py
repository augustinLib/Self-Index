import glob
import os

import numpy as np

from browsecomp.agent._paths import INDEXES
from selfindex.common.io import read_pickle
from selfindex.common.models.roles import encoder_model
from selfindex.encoding.contracts import MappedCorpusEmbeddingBackend
from selfindex.runtime.platform.devices import compute_device
from selfindex.runtime.platform.environment import ensure_env

MAX_LENGTH = 8192
INDEX_DIR = os.path.join(INDEXES, "qwen3-embedding-8b")


def dense_encoder_options(device, cache_dir):
    import torch

    return {
        "pooling": "eos",
        "normalize": True,
        "lora_name_or_path": None,
        "cache_dir": cache_dir,
        "torch_dtype": torch.float32 if device == "cpu" else torch.float16,
        "attn_implementation": "sdpa",
    }


class QwenEmbBackend(MappedCorpusEmbeddingBackend):

    QUERY_PREFIX = (
        "Instruct: Given a web search query, retrieve relevant passages that answer the "
        "query\nQuery:"
    )

    def __init__(self):
        self._model = self._tok = None
        self._vectors = self._lookup = None

    def doc_emb(self):
        if self._vectors is None:
            reps, lookup = [], []
            for path in sorted(glob.glob(os.path.join(INDEX_DIR, "corpus.shard*.pkl"))):
                vectors, ids = read_pickle(path)
                if np.asarray(vectors).ndim != 2 or len(vectors) != len(ids):
                    raise ValueError("embedding shard rows and document IDs differ")
                reps.append(np.asarray(vectors, np.float16))
                lookup.extend(ids)
            if not reps:
                raise ValueError("no corpus embedding shards found")
            if len(set(map(str, lookup))) != len(lookup):
                raise ValueError("embedding shard document IDs must be unique")
            self._vectors, self._lookup = (
                np.concatenate(reps),
                [str(d) for d in lookup],
            )
        return self._vectors

    @property
    def lookup(self):
        self.doc_emb()
        return self._lookup

    def _load_model(self):
        if self._model is None:
            import torch
            from tevatron.retriever.driver.encode import DenseModel
            from transformers import AutoTokenizer

            torch.backends.cuda.enable_cudnn_sdp(False)
            device = compute_device()
            hf_home = ensure_env().hf_home
            self._model = (
                DenseModel.load(
                    encoder_model(), **dense_encoder_options(device, hf_home)
                )
                .to(device)
                .eval()
            )
            self._tok = AutoTokenizer.from_pretrained(
                encoder_model(),
                cache_dir=hf_home,
                padding_side="left",
                truncation_side="right",
            )

    def _encode(self, texts, batch, query):
        import torch

        self._load_model()
        out = []
        for start in range(0, len(texts), batch):
            batch_texts = (
                [self.QUERY_PREFIX + t for t in texts[start : start + batch]]
                if query
                else texts[start : start + batch]
            )
            inputs = self._tok(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(compute_device()) for k, v in inputs.items()}
            with (
                torch.no_grad(),
                torch.amp.autocast("cuda", enabled=compute_device() != "cpu"),
            ):
                encoded = (
                    self._model.encode_query(inputs)
                    if query
                    else self._model.encode_passage(inputs)
                )
            out.append(encoded.float().cpu().numpy())
        vectors = np.concatenate(out).astype(np.float32)
        bad = np.isnan(vectors).any(axis=1)
        if bad.any():
            vectors[bad] = 0
            print(f"[qwen-emb] {int(bad.sum())} NaN rows zeroed", flush=True)
        return vectors

    def encode_queries(self, texts):
        return self._encode(texts, 16, query=True)

    def encode_passages(self, texts):
        return self._encode(texts, 256, query=False)

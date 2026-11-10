from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from selfindex.common.models.encoders import ENCODERS
from selfindex.encoding.contracts import TextEncoder
from selfindex.index.contracts import KeyIndex
from selfindex.retrieval.contracts import Retriever
from selfindex.runtime.registry import Registry


class RetrieverBackend(ABC):
    name: str
    has_embeddings = False

    def encode(self, texts, **options):
        if not self.has_embeddings:
            raise TypeError(f"{type(self).__name__} does not provide embeddings")
        from selfindex.encoding.encoders import encode

        return encode(texts, **options)

    @abstractmethod
    def build(self, texts, encode): ...

    def create(self, texts):
        pair = self.build(texts, self.encode)
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError("backend build() must return (KeyIndex, Retriever)")
        index, retriever = pair
        if not isinstance(index, KeyIndex) or not isinstance(retriever, Retriever):
            raise TypeError("backend build() must return (KeyIndex, Retriever)")
        if index.N != len(texts):
            raise ValueError("backend index size does not match the corpus")
        if index.has_embeddings != self.has_embeddings:
            raise ValueError("backend and index embedding capabilities disagree")
        return index, retriever


class DenseBackend(RetrieverBackend):
    name = "dense"
    has_embeddings = True

    def __init__(self, encoder=None, *, name=None):
        if encoder is not None and not isinstance(encoder, TextEncoder):
            raise TypeError("encoder must implement TextEncoder")
        self.encoder = encoder
        if name is not None:
            self.name = name

    def encode(self, texts, **options):
        if self.encoder is None:
            return super().encode(texts, **options)
        is_query = options.pop("is_query", False)
        if options:
            raise TypeError(f"unsupported encoder options: {sorted(options)}")
        method = (
            self.encoder.encode_queries if is_query else self.encoder.encode_passages
        )
        return method(texts)

    def build(self, texts, encode):
        from selfindex.index.dense import IndexState
        from selfindex.retrieval.dense import DenseRetriever

        index = IndexState(texts, encode)
        return index, DenseRetriever(index, encode)


class BM25Backend(RetrieverBackend):
    name = "bm25"

    def __init__(self, entries="all", *, name=None):
        self.entries = entries
        if name is not None:
            self.name = name

    def build(self, texts, encode):
        from selfindex.index.lexical import LexicalIndexAdapter
        from selfindex.retrieval.lexical import LexicalRetriever

        retriever = LexicalRetriever(texts, self.entries)
        return LexicalIndexAdapter(retriever), retriever


@dataclass(frozen=True)
class RetrieverSpec:

    name: str
    backend: str
    encoder: str | None = None


BACKENDS: Registry[RetrieverBackend] = Registry("retriever backend")
RETRIEVERS: Registry[RetrieverSpec] = Registry("retriever")


def register_retriever(backend: RetrieverBackend, *, name=None, encoder=None):
    if not isinstance(backend, RetrieverBackend):
        raise TypeError("backend must implement RetrieverBackend")
    name = backend.name if name is None else name
    RETRIEVERS.validate_name(name)
    if backend.name in BACKENDS and BACKENDS[backend.name] is not backend:
        raise ValueError(f"retriever backend already registered: {backend.name}")
    if encoder is not None and (encoder not in ENCODERS or not backend.has_embeddings):
        raise ValueError(f"{encoder} is not an encoder of an embedding backend")
    if backend.name not in BACKENDS:
        BACKENDS.validate_name(backend.name)
        BACKENDS.register(backend.name, backend)
    return RETRIEVERS.register(name, RetrieverSpec(name, backend.name, encoder))


_dense = DenseBackend()
register_retriever(BM25Backend())
register_retriever(BM25Backend("joined", name="bm25-concat"))
register_retriever(_dense, name="bge", encoder="bge")
register_retriever(_dense, name="qwen3", encoder="qwen3")


def get_backend(name):
    if name in RETRIEVERS:
        name = RETRIEVERS[name].backend
    if name not in BACKENDS:
        raise ValueError(
            f"unknown retriever backend: {name}; available: {', '.join(BACKENDS)}"
        )
    return BACKENDS[name]

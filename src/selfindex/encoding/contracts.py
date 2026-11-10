from abc import ABC, abstractmethod

import numpy as np


class TextEncoder(ABC):
    @abstractmethod
    def encode_queries(self, texts) -> np.ndarray: ...

    @abstractmethod
    def encode_passages(self, texts) -> np.ndarray: ...


class CorpusEmbeddingBackend(TextEncoder):
    @abstractmethod
    def doc_emb(self) -> np.ndarray: ...


class MappedCorpusEmbeddingBackend(CorpusEmbeddingBackend):
    @property
    @abstractmethod
    def lookup(self): ...

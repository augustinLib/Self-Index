from __future__ import annotations

import numpy as np

from selfindex.runtime.platform import settings as C

_ANALYZER = None


def analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        from pyserini import analysis

        _ANALYZER = analysis.Analyzer(analysis.get_lucene_analyzer())
    return _ANALYZER


class BM25Index:

    def __init__(self, texts=None, corpus=None):
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity

        if corpus is None:
            lucene = analyzer()
            corpus = [lucene.analyze(text) for text in (texts or [])]
        self.corpus = corpus
        self.n_docs = len(corpus)
        if not self.n_docs:
            self.dic = self.model = self.index = None
            return
        self.dic = Dictionary(corpus)
        self.model = LuceneBM25Model(dictionary=self.dic, k1=C.BM25_K1, b=C.BM25_B)
        self.index = SparseMatrixSimilarity(
            self.model[list(map(self.dic.doc2bow, corpus))],
            num_docs=self.n_docs,
            num_terms=max(len(self.dic), 1),
            normalize_queries=False,
            normalize_documents=False,
        )

    def score_tokens(self, query_tokens):
        if not self.n_docs:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(
            self.index[self.model[self.dic.doc2bow(query_tokens)]], dtype=np.float32
        )

import numpy as np

from selfindex.storage.artifacts import LocalArtifactStore


class PipelineEvaluation:
    def __init__(
        self, corpus, retriever, embeddings, golds, exclusions, out, *, metric
    ):
        self.metric = metric
        self.queries = corpus.queries
        self.texts = [query["query"] for query in self.queries]
        self.retriever = retriever
        self.embeddings = embeddings
        self.golds = golds
        self.exclusions = exclusions
        self.store = LocalArtifactStore(out)
        self.base, self.base_per_query = self.measure() if self.texts else (0.0, [])

    @property
    def enabled(self):
        return bool(self.texts)

    def measure(self):
        values, offset = [], 0
        for rows in self.retriever.score_query_batches(self.texts, self.embeddings):
            rows = self._checked_block(rows)
            end = offset + len(rows)
            if end > len(self.texts):
                raise ValueError("evaluation received extra query scores")
            values.extend(self._block_values(rows, offset, end))
            offset = end
        if offset != len(self.texts):
            raise ValueError("evaluation did not receive all query scores")
        if not values:
            raise ValueError(
                "evaluation requires at least one query with gold documents"
            )
        return self.metric.scale * float(np.mean(values)), [
            self.metric.scale * value for value in values
        ]

    @staticmethod
    def _checked_block(rows):
        rows = np.asarray(rows)
        if rows.ndim != 2 or not len(rows):
            raise ValueError("evaluation requires nonempty 2D score blocks")
        if not np.isfinite(rows).all():
            raise ValueError("evaluation received non-finite scores")
        return rows

    def _block_values(self, rows, offset, end):
        return self.metric.values(
            rows, self.golds[offset:end], self.exclusions[offset:end]
        )

    def _write(self, name, payload):
        try:
            self.store.write_json(name, payload)
        except OSError as error:
            print(f"[EVAL] per-query record skipped ({error})", flush=True)

    def _scored_query_ids(self):
        return [
            query.get("qid", i) for i, query in enumerate(self.queries) if self.golds[i]
        ]

    def finish(self, domain):
        if not self.enabled:
            print(
                f"[EVAL] {domain}: no in-run evaluation "
                "(the benchmark scores the published keys)",
                flush=True,
            )
            return None
        score, per_query = self.measure()
        print(
            f"[EVAL] {domain} test {self.metric.label} "
            f"base={self.base:.2f} evolved={score:.2f} "
            f"delta={score - self.base:+.2f}",
            flush=True,
        )
        self._write(
            "per_query.json",
            {
                "qids": self._scored_query_ids(),
                "base": self.base_per_query,
                "evolved": per_query,
            },
        )
        return score

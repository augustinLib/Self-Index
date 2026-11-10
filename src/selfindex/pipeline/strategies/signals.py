from __future__ import annotations

import json
import os
import random
import time

import numpy as np

from selfindex.pipeline.contracts import Context, Signals, SignalSource
from selfindex.pipeline.exploration.filters import filter_queries
from selfindex.pipeline.settings import SignalSettings


def _add_phase_seconds(ctx, phase, seconds):
    phase_seconds = ctx.round_state.phase_seconds
    phase_seconds[phase] = phase_seconds.get(phase, 0.0) + seconds


class QuerySimulator(SignalSource):

    name = "query-simulator"

    def __init__(self):
        self.settings = SignalSettings.from_env()

    def _gen_queries(self, ctx, docs):
        queries, sources = [], []
        by_doc = ctx.gen.generate_batch(docs)
        for doc, doc_queries in by_doc.items():
            for query in doc_queries:
                queries.append(query)
                sources.append(doc)
        return queries, np.array(sources)

    @staticmethod
    def _query_file(ctx):
        round_index = ctx.round_state.round
        return os.path.join(
            ctx.run_out,
            "queries.jsonl" if round_index == 0 else f"queries_r{round_index}.jsonl",
        )

    @staticmethod
    def _write_query_records(stream, queries, sources):
        for query, source in zip(queries, sources):
            record = {"q": query, "s": int(source)}
            record_line = json.dumps(record, ensure_ascii=False) + "\n"
            stream.write(record_line)

    def _filter(self, ctx, queries, sources, accepted=()):
        pairs = list(zip([int(x) for x in sources], queries))
        kept, stats = filter_queries(
            ctx.llm,
            ctx.texts,
            pairs,
            ctx.query_state.used_before(ctx.round_state.round) + list(accepted),
        )
        _add_phase_seconds(ctx, "answerability", float(stats.get("t_answer", 0.0)))
        _add_phase_seconds(ctx, "dissimilarity", float(stats.get("t_dissim", 0.0)))
        print(
            f"[explore] candidates {stats['candidates']:,} | answerable {stats['answerable']:,} | "
            f"duplicate {stats['duplicate']:,} | accepted {stats['kept']:,}",
            flush=True,
        )
        return [q for _, q in kept], np.array([d for d, _ in kept], np.int64)

    def _round_queries(self, ctx, order):
        round_index = ctx.round_state.round
        if (rounds := ctx.query_state.rounds) is None:
            rounds = ctx.query_state.rounds = {}
        path = self._query_file(ctx)
        queries, sources = self._draw_queries(ctx, order)
        rounds[round_index] = {"Q": queries, "SRC": sources}
        try:
            self._write_query_file(path, queries, sources)
        except Exception as ex:
            print(f"[warn] query file not written: {ex}", flush=True)
        return queries, sources

    def _generate_and_filter(self, ctx, docs, accepted=()):
        started = time.time()
        queries, sources = self._gen_queries(ctx, docs)
        _add_phase_seconds(ctx, "query_gen", time.time() - started)
        started = time.time()
        queries, sources = self._filter(ctx, queries, sources, accepted=accepted)
        _add_phase_seconds(ctx, "query_filter", time.time() - started)
        return queries, sources

    def _draw_queries(self, ctx, order):
        limit = self.settings.queries_per_round
        queries, sources, start = [], [], 0
        while len(queries) < limit and start < len(order):
            batch = order[start : start + limit - len(queries)]
            start += len(batch)
            texts, origins = self._generate_and_filter(ctx, batch, queries)
            queries.extend(texts)
            sources.extend(int(source) for source in origins)
        print(
            f"[explore] {len(queries):,} of {limit:,} queries from "
            f"{len(set(sources)):,} source documents",
            flush=True,
        )
        return queries, np.array(sources, np.int64)

    def _write_query_file(self, path, queries, sources):
        with open(path, "w") as fh:
            self._write_query_records(fh, queries, sources)

    def _source_order(self, ctx):
        order = list(range(len(ctx.texts)))
        random.Random(f"{self.settings.seed}:{ctx.round_state.round}").shuffle(order)
        return order

    def build(self, ctx: Context) -> Signals:
        queries, _ = self._round_queries(ctx, self._source_order(ctx))
        return Signals(meta={"queries": list(queries)})

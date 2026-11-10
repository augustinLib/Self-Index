from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from selfindex.evaluation.evaluator import PipelineEvaluation
from selfindex.evaluation.metrics import NDCGMetric
from selfindex.pipeline.diagnostics import write_round_log
from selfindex.pipeline.engine import Context, OptimizerLoop
from selfindex.pipeline.exploration.simulator import QueryGenerator
from selfindex.pipeline.lifecycle.preparation import _eligible
from selfindex.pipeline.presets import PRESET, build_slots
from selfindex.pipeline.settings import (
    PARAMETERS,
    LoopSettings,
    PipelineSettings,
    parameter,
)
from selfindex.runtime.catalog import retrievers
from selfindex.settings import DOCUMENT_CLIP_CHARS, KEY_MAX_CHARS
from selfindex.storage.artifacts import LocalArtifactStore
from selfindex.storage.keys import publish_keys, write_keys
from selfindex.storage.provenance import ProvenanceWriter


class _TsWriter:

    def __init__(self, raw):
        self._raw = raw
        self._at_line_start = True

    def write(self, text):
        out = []
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part and self._at_line_start:
                out.append(time.strftime("[%H:%M:%S] ") + part)
            else:
                out.append(part)
            if i < len(parts) - 1:
                out.append("\n")
                self._at_line_start = True
            elif part:
                self._at_line_start = False
        return self._raw.write("".join(out))

    def flush(self):
        return self._raw.flush()

    def __getattr__(self, name):
        return getattr(self._raw, name)


def _install_ts_stdout():
    if not isinstance(sys.stdout, _TsWriter):
        sys.stdout = _TsWriter(sys.stdout)


def _settings_banner():
    changed = [
        f"{key}={os.environ[key]} (default {value})"
        for key, value in PARAMETERS.items()
        if os.environ.get(key, str(value)) != str(value)
    ]
    print(
        (
            "[0] settings changed from the defaults: " + " | ".join(changed)
            if changed
            else "[0] default settings"
        ),
        flush=True,
    )


def _snapshot_sources(out):
    ProvenanceWriter().write(out)


def _write_summary(out, preset, base, evolved, report, ctx, backend_name):
    env_snap = {k: os.environ[k] for k in PARAMETERS if k in os.environ}
    n_docs = len(ctx.texts)
    store = LocalArtifactStore(out)
    record = {
        "preset": preset,
        "base": base,
        "evolved": evolved,
        "report": report,
        "env": env_snap,
        "eligible": len(ctx.eligible),
        "retriever": backend_name,
        "achieved_cov": report.get("edited_docs", 0) / n_docs if n_docs else None,
    }
    store.write_json("summary.json", record, indent=1)


def _query_labels(queries):
    golds = [set(int(g) for g in q.get("gold_idxs", [])) for q in queries]
    excluded = [set(int(x) for x in (q.get("excluded_idxs") or [])) for q in queries]
    return golds, excluded


def _select_encoder(retriever, corpus):
    from selfindex.runtime.platform.settings import EncoderSettings, use_encoder

    if retriever.encoder is None:
        return
    settings = use_encoder(
        EncoderSettings.of(retriever.encoder, **corpus.encoder_settings)
    )
    if corpus.encoder_settings:
        print(
            f"[0] {corpus.name}: encoder {settings.name} with the corpus's "
            f"{', '.join(sorted(corpus.encoder_settings))}",
            flush=True,
        )


def _load_task(dom, retriever):
    from selfindex.runtime.catalog.benchmarks import load_corpus

    corpus = load_corpus(dom)
    _select_encoder(retriever, corpus)
    texts = corpus.texts
    golds, excluded = _query_labels(corpus.queries)
    backend = retrievers.get_backend(retriever.backend)
    query_embeddings = (
        backend.encode([q["query"] for q in corpus.queries], is_query=True)
        if corpus.queries and backend.has_embeddings
        else None
    )
    print(
        f"[0] {dom}: corpus={len(texts)} test-q={sum(1 for g in golds if g)}",
        flush=True,
    )
    return corpus, texts, golds, excluded, query_embeddings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="selfindex.pipeline.run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("domain", help="corpus to evolve (a registered corpus name)")
    parser.add_argument("tag", help="run tag")
    parser.add_argument(
        "--retriever",
        required=True,
        choices=sorted(retrievers.RETRIEVERS),
        help="registered retriever the index is evolved for",
    )
    parser.add_argument(
        "--keys-out",
        metavar="PATH",
        help="also publish the final keys to PATH, one record (k, docid) per key",
    )
    return parser.parse_args(argv)


def _build_context(texts, llm, backend, out):
    index, retriever = backend.create(texts)
    query_generator = QueryGenerator(llm, texts)
    eligible_docs = _eligible(texts, llm=llm)
    print(f"[0] screening: eligible={len(eligible_docs)}", flush=True)
    ctx = Context(
        texts=texts,
        encode=backend.encode,
        llm=llm,
        index=index,
        retriever=retriever,
        gen=query_generator,
        eligible=eligible_docs,
        run_out=str(out),
    )
    return ctx


def _record_validation_stats(loop, report):
    if loop.validation:
        report["validation"] = {
            name: int(value) for name, value in sorted(loop.validation.items())
        }
        print(f"[validation] {report['validation']}", flush=True)


def _output_dir(dom, tag):
    from selfindex.runtime.catalog.benchmarks import runs_directory

    out = Path(runs_directory(dom)) / f"{dom}_pipe_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fingerprint(texts):
    digest = hashlib.sha256()
    for text in texts:
        data = (text or "").encode("utf-8", "surrogatepass")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _model_identity(llm, retriever):
    from selfindex.runtime.platform.settings import encoder_settings

    encoder = encoder_settings() if retriever.encoder is not None else None
    return {
        "llm": str(getattr(llm, "model", "") or ""),
        "encoder": asdict(encoder) if encoder is not None else None,
    }


def _run_identity(names, settings, loop_settings, metric, corpus, llm, retriever):
    record = {
        "schema_version": 1,
        "domain": names["domain"],
        "tag": names["tag"],
        "preset": names["preset"],
        "settings": asdict(settings),
        "framework": {"key_max_chars": KEY_MAX_CHARS, "doc_clip": DOCUMENT_CLIP_CHARS},
        "loop": asdict(loop_settings),
        "parameters": {name: parameter(name) for name in PARAMETERS},
        "evaluation": {
            "name": metric.name,
            "label": metric.label,
            "scale": metric.scale,
        },
    }
    record["corpus"] = _fingerprint(corpus.texts)
    record["document_ids"] = _fingerprint(list(map(str, corpus.ids)))
    record["model_identity"] = _model_identity(llm, retriever)
    return record


def _write_run_config(out, record):
    _snapshot_sources(out)
    LocalArtifactStore(out).write_json("run_config.json", record, indent=2)


def _build_loop(ctx, slots, loop_settings):
    return OptimizerLoop(ctx, *slots, settings=loop_settings)


def _print_slots(preset, slots):
    print(
        f"[0] {preset}: signal={slots.signal.name}",
        flush=True,
    )


def _default_llm():
    from selfindex.llm.client import LLM

    return LLM()


def _evaluation(corpus, ctx, query_embeddings, golds, excluded, out, metric):
    evaluation = PipelineEvaluation(
        corpus, ctx.retriever, query_embeddings, golds, excluded, out, metric=metric
    )
    if not evaluation.enabled:
        print(
            "[EVAL] no queries: externally scored corpus, in-run evaluation skipped",
            flush=True,
        )
    return evaluation


def _write_loop_outputs(out, loop, report):
    _record_validation_stats(loop, report)
    write_round_log(out / "round_log.json", getattr(loop, "round_log", None))


def _finish_run(out, names, ctx, corpus, evaluation, report, keys_out):
    evolved = evaluation.finish(names["domain"])
    print(f"[EVAL] run: {report}", flush=True)
    write_keys(out / "keys.jsonl", ctx.index.generated_keys)
    _write_summary(
        out,
        names["preset"],
        evaluation.base,
        evolved if evaluation.enabled else None,
        report,
        ctx,
        names["backend"],
    )
    if keys_out:
        if len(corpus.ids) != ctx.index.N:
            raise ValueError("document IDs do not match the index corpus")
        publish_keys(ctx.index.generated_keys, corpus.ids, keys_out)


def main(argv=None, *, llm=None):
    _install_ts_stdout()
    args = parse_args(argv)
    dom, preset, tag = args.domain, PRESET, args.tag
    retriever = retrievers.RETRIEVERS[args.retriever]
    backend = retrievers.get_backend(retriever.backend)
    settings = PipelineSettings.from_env(retriever.name)
    metric = NDCGMetric()
    out = _output_dir(dom, tag)
    llm = _default_llm() if llm is None else llm
    corpus, texts, golds, excluded, query_embeddings = _load_task(dom, retriever)
    loop_settings = LoopSettings.from_env()
    names = {"domain": dom, "tag": tag, "preset": preset, "backend": backend.name}
    identity = _run_identity(
        names, settings, loop_settings, metric, corpus, llm, retriever
    )
    _write_run_config(out, identity)
    ctx = _build_context(texts, llm, backend, out)
    evaluation = _evaluation(
        corpus, ctx, query_embeddings, golds, excluded, out, metric
    )
    slots = build_slots()
    loop = _build_loop(ctx, slots, loop_settings)
    _print_slots(preset, slots)
    _settings_banner()
    report = loop.run()
    _write_loop_outputs(out, loop, report)
    _finish_run(out, names, ctx, corpus, evaluation, report, args.keys_out)

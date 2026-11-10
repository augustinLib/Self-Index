from __future__ import annotations

import hashlib
import multiprocessing
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from selfindex.pipeline.settings import parameter

_URL_RE = re.compile(r"https?://\S+|doi\.org\S*|10\.\d{4,}/\S+|\[.*?\]\(.*?\)|!\[.*?\]")

_INFO_TEXTS = None
RECORD_SAMPLE_SIZE = 200
RECORD_MIN_RATE = 0.5
RECORD_DOCUMENT_CHARS = 2000
MIN_INFO_WORDS = 60
MIN_DOCUMENT_CHARS = 20


def _info_words(t):
    text = _URL_RE.sub(" ", t or "")
    text = re.sub(r"[\\{}$_^&%#]|begin|end|section|Tags:", " ", text)
    codes = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
    if codes.size < 2:
        return 0
    letter = ((codes >= 65) & (codes <= 90)) | ((codes >= 97) & (codes <= 122))
    prev = np.empty_like(letter)
    prev[0] = False
    prev[1:] = letter[:-1]
    return int(np.count_nonzero(letter[:-1] & letter[1:] & ~prev[:-1]))


def _info_words_span(span):
    start, end = span
    return np.fromiter(
        (_info_words(t) for t in _INFO_TEXTS[start:end]), np.int32, end - start
    )


def _info_words_all(texts):
    n = len(texts)
    workers = min(8, os.cpu_count() or 4)
    parallel = n >= 5000 and workers > 1
    if parallel and (counts := _parallel_info_words(texts, workers)) is not None:
        return counts
    return np.fromiter((_info_words(t) for t in texts), np.int32, n)


def _parallel_info_words(texts, workers):
    global _INFO_TEXTS
    n = len(texts)
    _INFO_TEXTS = texts
    try:
        started = time.time()
        n_spans = workers * 4
        spans = [
            (round(i * n / n_spans), round((i + 1) * n / n_spans))
            for i in range(n_spans)
        ]
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        ) as executor:
            parts = list(executor.map(_info_words_span, spans))
        counts = np.concatenate(parts)
        print(
            f"[0] information counts computed once: {n:,} docs, {workers} cores, {time.time() - started:.0f}s",
            flush=True,
        )
        return counts
    except Exception as err:
        print(f"[0] parallel information count failed ({err}), single core", flush=True)
        return None
    finally:
        _INFO_TEXTS = None


def _unique_documents(texts, accept):
    seen, kept = set(), []
    for doc, raw in enumerate(texts):
        text = (raw or "").strip()
        digest = hashlib.md5(text.lower().encode()).hexdigest()
        if len(text) >= MIN_DOCUMENT_CHARS and digest not in seen and accept(doc, raw):
            seen.add(digest)
            kept.append(doc)
    return kept


RECORD_PROMPT = """Classify the document's dominant surface form.
- RECORD: primarily a structured data artifact: a database schema, table or column listing, \
field/value dump, configuration block, or similar machine-oriented enumeration.
- PROSE: primarily natural-language text written for human readers: articles, discussions, \
explanations, problem statements, or code accompanied by narrative context.

Answer with exactly one word: RECORD or PROSE.

DOCUMENT:
<<<DOC_START>>>
{doc}
<<<DOC_END>>>
"""


def judge_record(llm, texts, docs):
    jobs = [
        [
            {
                "role": "user",
                "content": RECORD_PROMPT.format(doc=texts[d][:RECORD_DOCUMENT_CHARS]),
            }
        ]
        for d in docs
    ]
    raws = llm.batch(jobs, temperature=0.6, max_tokens=16)
    return [
        int(d)
        for d, raw in zip(docs, raws)
        if isinstance(raw, str) and re.match(r"\s*RECORD\b", raw, re.I)
    ]


def _judged_records(texts, llm):
    failed = [
        doc
        for doc, text in enumerate(texts)
        if len((text or "").strip()) >= MIN_DOCUMENT_CHARS
        and _info_words(text) < MIN_INFO_WORDS
    ]
    if not failed:
        return set()
    sample = random.Random(parameter("SEED")).sample(
        failed, min(RECORD_SAMPLE_SIZE, len(failed))
    )
    rate = len(judge_record(llm, texts, sample)) / len(sample)
    print(
        f"[0] record judge sample: RECORD rate {rate:.0%} of {len(failed):,} below the information floor",
        flush=True,
    )
    if rate < RECORD_MIN_RATE:
        return set()
    records = set(judge_record(llm, texts, failed))
    print(f"[0] record judge: {len(records):,} records are eligible", flush=True)
    return records


def _eligible(texts, llm):
    records = _judged_records(texts, llm)
    return _unique_documents(
        texts, lambda doc, text: _info_words(text) >= MIN_INFO_WORDS or doc in records
    )

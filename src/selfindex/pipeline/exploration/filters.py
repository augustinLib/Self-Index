from __future__ import annotations

import re
import time as _time
from collections import Counter

from selfindex.pipeline.exploration.dissimilarity import dissimilar_mask
from selfindex.common.judge_prompts import ANSWERABILITY, PASSING_SCORE


def answerable(llm, texts, pairs):
    if not pairs:
        return []
    jobs = [
        [
            {
                "role": "user",
                "content": ANSWERABILITY.format(doc=texts[d][:DOCUMENT_CHARS], query=q),
            }
        ]
        for d, q in pairs
    ]
    raws = llm.batch(
        jobs, temperature=ANSWERABILITY_TEMPERATURE, max_tokens=ANSWERABILITY_TOKENS
    )
    return [_answer_score(raw) >= PASSING_SCORE for raw in raws]


DOCUMENT_CHARS = 2400
ANSWERABILITY_TOKENS = 32
ANSWERABILITY_TEMPERATURE = 0.3


def _answer_score(raw):
    match = re.match(r"\s*([0-3])\b", str(raw or ""))
    return int(match.group(1)) if match else 0


def filter_queries(llm, texts, pairs, used_texts):
    stats = Counter(candidates=len(pairs))
    started = _time.time()
    ok = answerable(llm, texts, pairs)
    stats["t_answer"] = _time.time() - started
    started = _time.time()
    stats["answerable"] = sum(ok)
    candidates = [(d, q) for (d, q), accepted in zip(pairs, ok) if accepted]
    novel = dissimilar_mask(list(used_texts), [q for _, q in candidates])
    kept = [pair for pair, is_novel in zip(candidates, novel) if is_novel]
    stats["duplicate"] = len(candidates) - len(kept)
    stats["t_dissim"] = _time.time() - started
    stats["kept"] = len(kept)
    return kept, stats

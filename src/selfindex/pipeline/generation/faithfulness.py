from __future__ import annotations

import re

from selfindex.common.judge_prompts import PASSING_SCORE

FAITHFULNESS_TOKEN_OVERHEAD = 20

FAITHFULNESS_TOKENS_PER_KEY = 32

_SCORE_LINE = re.compile(r"\s*\[?(\d+)\]?\s*:.*=>\s*([0-3])\b")


def parse_faithfulness(raw: str, n: int) -> set[int]:

    scores = {}
    for line in str(raw).splitlines():
        if (match := _SCORE_LINE.match(line)) is None:
            continue
        index, score = int(match.group(1)) - 1, int(match.group(2))
        if 0 <= index < n:
            scores.setdefault(index, score)
    return {index for index, score in scores.items() if score >= PASSING_SCORE}

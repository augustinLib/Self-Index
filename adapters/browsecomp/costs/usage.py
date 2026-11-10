import glob
import os
from typing import NamedTuple

from selfindex.common.io import read_json
from selfindex.common.models.roles import (
    GEMINI_AGENT_MODEL,
    KIMI_AGENT_MODEL,
    NANO_AGENT_MODEL,
)

PRICES = {
    NANO_AGENT_MODEL: (0.20, 0.02, 1.25),
    GEMINI_AGENT_MODEL: (0.75, 0.075, 3.75),
    KIMI_AGENT_MODEL: (0.45, 0.07, 2.25),
}


class _RunUsage(NamedTuple):
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost: float


def _token_count(usage, key):
    return int(usage.get(key) or 0)


def _run_usage(usage, prices):
    input_price, cached_price, output_price = prices
    input_tokens = _token_count(usage, "input_tokens")
    cached = _token_count(usage, "input_tokens_cached")
    output = _token_count(usage, "output_tokens")
    reasoning = _token_count(usage, "included_reasoning_tokens")
    cost = (
        (input_tokens - cached) * input_price
        + cached * cached_price
        + output * output_price
    ) / 1e6
    return _RunUsage(input_tokens, cached, output, reasoning, cost)


def usage_summary(run_dir, prices):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "run_*.json"))):
        try:
            record = read_json(path)
        except Exception:
            continue
        if usage := record.get("usage") or {}:
            rows.append(_run_usage(usage, prices))
    if not rows:
        return {"n": 0, "cost": 0.0}
    return {
        "n": len(rows),
        "input": sum(row.input_tokens for row in rows),
        "cached": sum(row.cached_tokens for row in rows),
        "output": sum(row.output_tokens for row in rows),
        "reasoning": sum(row.reasoning_tokens for row in rows),
        "cost": sum(row.cost for row in rows),
    }


def print_usage(run_dir, prices):
    summary = usage_summary(run_dir, prices)
    if not (n := summary["n"]):
        print(f"no usage: {run_dir}")
        return
    print(
        f"total {n} queries · in {summary['input']:,} (cache {summary['cached']:,}) · "
        f"out {summary['output']:,} · ${summary['cost']:.4f} "
        f"(per query ${summary['cost'] / n:.4f})",
        flush=True,
    )

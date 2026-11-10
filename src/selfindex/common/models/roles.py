from __future__ import annotations

from selfindex.common.models.encoders import ENCODERS

AGENT_MODEL = "openai/gpt-oss-120b"
NANO_AGENT_MODEL = "gpt-5.4-nano-2026-03-17"
GEMINI_AGENT_MODEL = "gemini-3.7-flash"
KIMI_AGENT_MODEL = "moonshotai/kimi-k2.5"
LME_EVALUATOR = "gpt-5.2"
COUNT_TOKENIZER = "Qwen/Qwen3-0.6B"


def encoder_model():
    return ENCODERS["qwen3"]["model"]

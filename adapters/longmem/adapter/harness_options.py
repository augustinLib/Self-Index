import os
from pathlib import Path
from types import MappingProxyType

from longmem.adapter.paths import DATA_ROOT
from selfindex.common.models.roles import LME_EVALUATOR
from selfindex.llm.backbones import JudgeBackbone
from selfindex.runtime.platform.endpoints import Endpoint

READER_PROTOCOL = MappingProxyType(
    {
        "temperature": 0.6,
        "top-p": 0.95,
        "top-k": 20,
        "max-completion-tokens": 20000,
        "memory-context-max-tokens": 200000,
        "reader-max-concurrent-requests": 96,
        "prompt-build-max-workers": 1,
    }
)
TIER = "small"
DOMAINS = ("web", "enterprise")
MEMORY_TEMPLATES = ("rag_query_to_slice", "rag_query_to_slice_notes", "agentrunbook_r")


def harness_arguments(domain, out):
    for role in ("READER", "JUDGE"):
        if not os.environ.get(f"{role}_BASE_URL"):
            raise ValueError(f"configure {role}_BASE_URL")
    reader = Endpoint(os.environ["READER_BASE_URL"])
    judge = Endpoint(os.environ["JUDGE_BASE_URL"], LME_EVALUATOR)
    evaluator = JudgeBackbone.for_model(judge.model)
    out = Path(out)
    options = {
        "domain": domain,
        "questions-path": out / "runtime_inputs/questions.json",
        "haystack-path": out / "runtime_inputs/haystack.json",
        "trajectories-path": DATA_ROOT / "trajectories.jsonl",
        "memory-config-path": out / "runtime_inputs/memory_config.json",
        "output-dir": out,
        "model": reader.model,
        "base-url": reader.base_url,
        "api-key-env": reader.api_key_env,
        **READER_PROTOCOL,
        "evaluator-model": evaluator.model,
        "evaluator-base-url": judge.base_url,
        "evaluator-api-key-env": judge.api_key_env,
    }
    if evaluator.reasoning_effort is not None:
        options["evaluator-reasoning-effort"] = evaluator.reasoning_effort
    return [
        part for name, value in options.items() for part in (f"--{name}", str(value))
    ]

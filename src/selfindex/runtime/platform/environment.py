from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[4])
ADAPTERS_ROOT = os.path.join(REPO_ROOT, "adapters")
EXTERNALS_ROOT = os.path.join(REPO_ROOT, "externals")
IMPORT_ROOTS = (os.path.join(REPO_ROOT, "src"), ADAPTERS_ROOT)
HF_HOME = os.path.join(REPO_ROOT, ".hf_home")
RETRIEVAL_RUNS_ROOT = os.path.join(REPO_ROOT, "runs")


@dataclass(frozen=True)
class Environment:
    repo_root: str
    hf_home: str


def resolve() -> Environment:
    return Environment(repo_root=REPO_ROOT, hf_home=HF_HOME)


_ENV: Environment | None = None


def ensure_env() -> Environment:
    global _ENV
    if _ENV is not None:
        return _ENV
    environment = resolve()
    _ENV = environment
    return environment

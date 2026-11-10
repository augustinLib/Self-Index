from pathlib import Path

from selfindex.runtime.platform.environment import EXTERNALS_ROOT

LONGMEM_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = Path(EXTERNALS_ROOT) / "LongMemEval-V2"
DATA_ROOT = UPSTREAM_ROOT / "data" / "longmemeval-v2"
OUTPUT_ROOT = LONGMEM_ROOT / "runs"


def keys_path(tag):
    return OUTPUT_ROOT / f"keys_{tag}.jsonl"


def evaluation_dir(tag, memory, domain):
    return OUTPUT_ROOT / tag / memory / domain

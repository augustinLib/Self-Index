from __future__ import annotations

import json
from pathlib import Path


def write_round_log(path, rounds):
    if not rounds:
        return
    with Path(path).open("w") as stream:
        json.dump(rounds, stream, ensure_ascii=False, indent=1)
    print(f"[round] round log written: {len(rounds)} rounds", flush=True)

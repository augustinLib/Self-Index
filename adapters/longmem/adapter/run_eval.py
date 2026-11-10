import argparse
import os
import subprocess
from pathlib import Path

from longmem.adapter.harness_options import DOMAINS, MEMORY_TEMPLATES
from selfindex.runtime.commands import subprocess_argv as cmd_argv
from selfindex.runtime.platform.environment import ensure_env


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="run_eval.py", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("domain", choices=DOMAINS)
    parser.add_argument("out_dir")
    parser.add_argument("keys_path")
    parser.add_argument(
        "memory", choices=MEMORY_TEMPLATES, help="official memory system"
    )
    return parser.parse_args(argv)


def _run_harness(root, domain, out, harness_args):
    Path(out).mkdir(parents=True, exist_ok=True)
    logfile = Path(out) / "evaluation.log"
    with open(logfile, "w") as log:
        returncode = subprocess.run(
            cmd_argv("lme.harness", harness_args),
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=root,
        ).returncode
    print(
        f"[selfindex-eval] {domain} {os.path.basename(out)} rc={returncode}", flush=True
    )
    return returncode


def main(argv=None):
    from longmem.adapter.harness_options import harness_arguments
    from longmem.adapter.inputs import LongMemInputs

    args = _parse_args(argv)
    root = Path(ensure_env().repo_root)
    out, keys = str(root / args.out_dir), str(root / args.keys_path)
    harness_args = harness_arguments(args.domain, out)
    LongMemInputs().prepare(args.domain, out, keys, args.memory)
    return _run_harness(root, args.domain, out, harness_args)

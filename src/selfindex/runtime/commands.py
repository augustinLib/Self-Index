from __future__ import annotations

import importlib
import inspect
import os
import sys

from selfindex.runtime.platform.environment import IMPORT_ROOTS, REPO_ROOT

COMMANDS = {
    "pipeline.launch": "selfindex.runtime.execution.launch:main",
    "retrieval.result": "selfindex.runtime.catalog.benchmarks:score_main",
    "repo.check": "selfindex.runtime.repository_check:main",
    "bc.pipeline": "browsecomp.pipeline.launch_pipeline:main",
    "bc.full": "browsecomp.pipeline.full_run:main",
    "bc.agent": "browsecomp.agent.run_backbone:main",
    "bc.client.oss": "browsecomp.agent.oss_client_ours:main",
    "bc.client.openai": "browsecomp.agent.openai_client_ours:main",
    "bc.client.or": "browsecomp.agent.or_client_ours:main",
    "bc.client.gemini": "browsecomp.agent.gemini_client_ours:main",
    "bc.mcp_server": "browsecomp.agent.mcp_server_ours:main",
    "bc.judge_run": "browsecomp.agent.judge_run:main",
    "bc.build_index": "browsecomp.pipeline.build_bm25_index:main",
    "bc.encode_keys": "browsecomp.pipeline.encode_keys:main",
    "lme.eval": "longmem.adapter.run_eval:main",
    "lme.harness": "longmem.adapter.harness_ours:main",
}


def register_command(name, target):
    if name in COMMANDS:
        raise ValueError(f"command already registered: {name}")
    if not name or not isinstance(target, str) or len(target.split(":")) != 2:
        raise ValueError("command target must be module:function")
    COMMANDS[name] = target


def resolve(name):
    if name not in COMMANDS:
        raise SystemExit(
            f"unknown command: {name}\nknown: {', '.join(sorted(COMMANDS))}"
        )
    module_name, function_name = COMMANDS[name].split(":")
    for root in reversed(IMPORT_ROOTS):
        if root not in sys.path:
            sys.path.insert(0, root)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"command {name} does not resolve to a callable")
    return function


def exit_code(result):
    return result if isinstance(result, int) and not isinstance(result, bool) else 0


def dispatch(name, argv):
    function = resolve(name)
    signature = inspect.signature(function)
    try:
        signature.bind(list(argv))
    except TypeError:
        signature.bind()
        saved_argv, sys.argv = sys.argv, [name] + list(argv)
        try:
            return exit_code(function())
        finally:
            sys.argv = saved_argv
    return exit_code(function(list(argv)))


def subprocess_argv(name, args=()):
    return [
        sys.executable,
        os.path.join(REPO_ROOT, "run.py"),
        "exec",
        name,
        *[str(arg) for arg in args],
    ]

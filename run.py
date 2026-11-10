#!/usr/bin/env python3
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _directory in ("adapters", "src"):
    if os.path.join(_ROOT, _directory) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _directory))

from selfindex.common.models.backbones import BACKBONES  # noqa: E402
from selfindex.runtime.catalog.retrievers import RETRIEVERS  # noqa: E402
from selfindex.runtime.runner import ExperimentSpec, Runner  # noqa: E402


def agent_models():
    from browsecomp.agent.backbones import agent_names

    return list(agent_names())


EXPERIMENT_OPTIONS = frozenset(
    {
        "--benchmark",
        "--retriever",
        "--agent-model",
        "--stage",
        "--tag",
        "--dry-run",
        "--llm-url",
    }
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_experiment_options(parser)
    _add_execution_options(parser)
    _add_subcommands(parser)
    return parser


def _add_experiment_options(parser):
    parser.add_argument(
        "--benchmark", metavar="NAME", help="benchmark to run (see: run.py list)"
    )
    parser.add_argument(
        "--retriever",
        choices=sorted(RETRIEVERS),
        help="frozen retriever (default: the benchmark's own)",
    )
    parser.add_argument(
        "--agent-model",
        help="BrowseComp-Plus search agent (default: gptoss; see: run.py list)",
    )
    parser.add_argument(
        "--stage",
        choices=["evolve", "evaluate", "full"],
        default="full",
        help="full: evolve the index and evaluate it; BrowseComp-Plus and "
        "LongMemEval-V2 also run evolve or evaluate alone; for BRIGHT and the table "
        "corpora, evaluate prints the scores of a finished run",
    )
    parser.add_argument(
        "--tag", help="run tag (default: derived from the configuration)"
    )


def _add_execution_options(parser):
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the execution plan without running models",
    )
    parser.add_argument(
        "--llm-url",
        help="optimizer LLM base URL (OpenAI-compatible; default: LLM_BASE_URL)",
    )


def _add_subcommands(parser):
    subparsers = parser.add_subparsers(dest="mode")
    list_parser = subparsers.add_parser("list", help="benchmarks and retrievers")
    list_parser.add_argument(
        "--commands", action="store_true", help="also list every registered step name"
    )
    exec_parser = subparsers.add_parser("exec", help="run one registered step")
    exec_parser.add_argument(
        "command", help="registered command name (see: run.py list --commands)"
    )
    exec_parser.add_argument("args", nargs=argparse.REMAINDER)


def cmd_list(show_commands=False):
    from selfindex.runtime.catalog.benchmarks import BENCHMARKS

    print("benchmarks (--benchmark NAME):")
    for name in sorted(BENCHMARKS):
        print(f"    {name:26s} {BENCHMARKS[name].describe()}")
    print("retrievers (--retriever NAME):", ", ".join(sorted(RETRIEVERS)))
    print("agents (--agent-model NAME):", ", ".join(agent_models()))
    print("optimizer backbones the paper reports (served at LLM_BASE_URL):")
    for name, model in sorted(BACKBONES.items()):
        print(f"    {name:26s} {model}")
    if show_commands:
        from selfindex.runtime.commands import COMMANDS

        print(f"registered steps (run.py exec <name>), {len(COMMANDS)} total:")
        for name in sorted(COMMANDS):
            print(f"    {name}")
    return 0


def _forwarded_args(args):
    return list(args[1:] if args[:1] == ["--"] else args)


def _validate_mode(parser, options, argv):
    if not options.mode:
        return
    prefix = argv[: argv.index(options.mode)]
    supplied = sorted({arg.split("=", 1)[0] for arg in prefix} & EXPERIMENT_OPTIONS)
    if supplied:
        parser.error(
            f"{', '.join(supplied)} cannot be used before {options.mode}; "
            "experiment options require --benchmark without a subcommand"
        )


def _run_mode(options):
    if options.mode == "list":
        return cmd_list(options.commands)
    from selfindex.runtime.commands import dispatch

    return dispatch(options.command, _forwarded_args(options.args))


def _runtime_env(parser, options):
    if not options.llm_url:
        return {}
    from selfindex.runtime.platform.endpoints import Endpoint

    try:
        urls = [Endpoint(url.strip()).base_url for url in options.llm_url.split(",")]
    except ValueError as error:
        parser.error(str(error))
    return {"LLM_BASE_URL": ",".join(urls)}


def _experiment_spec(options, runtime_env):
    return ExperimentSpec(
        benchmark=options.benchmark,
        retriever=options.retriever,
        agent_model=options.agent_model,
        stage=options.stage,
        tag=options.tag,
        runtime_env=runtime_env,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    options = parser.parse_args(argv)
    _validate_mode(parser, options, argv)
    if options.mode:
        return _run_mode(options)
    if not options.benchmark:
        parser.print_help()
        return 2
    spec = _experiment_spec(options, _runtime_env(parser, options))
    try:
        if options.dry_run:
            report = Runner().plan(spec).to_dict()
            report_json = json.dumps(report, indent=2)
            print(report_json)
            return 0
        return Runner().run_experiment(spec)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

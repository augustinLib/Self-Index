import os
import sys
from pathlib import Path

from selfindex.runtime.platform.endpoints import endpoint


def require_judge():
    return endpoint("JUDGE")


def main(argv):
    from browsecomp.adapters.judging import (
        JudgeSampling,
        RemoteAnswerJudge,
        load_evaluator,
    )

    connection = require_judge()
    instances = []

    def create_engine(**options):
        judge = RemoteAnswerJudge(connection)
        instances.append(judge)
        return judge

    evaluator = load_evaluator(create_engine, JudgeSampling)
    from browsecomp.adapters.paths import evaluation_root, prepare_evaluation_directory

    from browsecomp.pipeline.specification import input_files

    evaluator["mirror_directory_structure"] = prepare_evaluation_directory
    root = Path(__file__).resolve().parents[1]
    inputs = input_files()
    defaults = {
        "--eval_dir": str(evaluation_root()),
        "--ground_truth": str(inputs["ground_truth"]),
        "--qrel_evidence": str(inputs["qrel_evidence"]),
    }
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        arguments = list(argv)
        given = {arg.split("=", 1)[0] for arg in arguments}
        for option, value in defaults.items():
            if option not in given:
                arguments += [option, value]
        sys.argv = ["evaluate_run.py", *arguments]
        os.chdir(root)
        return evaluator["main"]()
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)
        for judge in instances:
            judge.close()

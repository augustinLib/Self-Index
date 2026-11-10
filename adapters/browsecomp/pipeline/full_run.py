import argparse
import os

from browsecomp.adapters.search_factory import SEARCH_ADAPTERS
from browsecomp.agent.backbones import agent_names, get_backbone
from browsecomp.pipeline.specification import FullRunPlan
from browsecomp.pipeline.workflow import BrowseCompWorkflow
from selfindex.runtime.plan import ExecutionPlan, StepSpec


class FullRun(FullRunPlan):
    def check_services(self, env):
        from browsecomp.agent.judge_run import require_judge
        from selfindex.runtime.configuration import validate_services

        validate_services(
            ExecutionPlan("bcplus", (StepSpec("browsecomp", "pipeline", [], env),))
        )
        require_judge()
        get_backbone(
            self.agent, retriever=self.retriever, keys=self.keys
        ).check_service()

    def execute(self, env, resume=False):
        return BrowseCompWorkflow(self, env, resume).execute()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retriever", required=True, choices=list(SEARCH_ADAPTERS))
    parser.add_argument("--agent", required=True, choices=agent_names())
    args = parser.parse_args(argv)
    environment = dict(os.environ)
    run = FullRun.create(args.tag, args.retriever, args.agent)
    run.check_services(environment)
    return run.execute(environment, resume=args.resume)

import os

from selfindex.runtime.factories import FACTORIES
from selfindex.runtime.platform.endpoints import Endpoint


def validate_services(plan):
    for step in plan.steps:
        env = {**os.environ, **step.env}
        for role in FACTORIES[step.family].required_services(step.task):
            if not (url := env.get(f"{role}_BASE_URL")):
                raise ValueError(
                    f"configure {role}_BASE_URL before running "
                    f"{step.family}/{step.task}"
                )
            for candidate in url.split(","):
                Endpoint(candidate.strip())

import importlib
import os
import sys

from browsecomp.agent._paths import UPSTREAM

UPSTREAM_PATHS = (
    UPSTREAM,
    os.path.join(UPSTREAM, "search_agent"),
    os.path.join(UPSTREAM, "searcher"),
)


def upstream_module(name):
    for path in UPSTREAM_PATHS:
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module(name)

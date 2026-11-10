import glob
import json
import os

RUN_PATTERN = "run_*.json"


def _answer(path):
    try:
        with open(path, encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not record.get("result"):
        return None
    return str(record.get("query_id"))


def answered_queries(directory):
    answers = map(_answer, glob.glob(os.path.join(directory, RUN_PATTERN)))
    return {query_id for query_id in answers if query_id is not None}


def discard_unanswered_runs(directory):
    unanswered = [
        path
        for path in glob.glob(os.path.join(directory, RUN_PATTERN))
        if _answer(path) is None
    ]
    for path in unanswered:
        os.remove(path)
    return len(unanswered)

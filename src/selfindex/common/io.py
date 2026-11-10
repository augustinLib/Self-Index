from __future__ import annotations

import json
import pickle


def read_lines(path, **kwargs):
    with open(path, **kwargs) as fh:
        yield from fh


def read_json(path, **kwargs):
    with open(path, **kwargs) as fh:
        return json.load(fh)


def write_json(path, obj, **kwargs):
    with open(path, "w") as fh:
        json.dump(obj, fh, **kwargs)


def read_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

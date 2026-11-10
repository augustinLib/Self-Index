from __future__ import annotations

from selfindex.common.json_response import extract_json


def extract_obj(raw) -> dict:
    parsed = extract_json(raw) if raw else None
    return parsed if isinstance(parsed, dict) else {}


def extract_list(raw, key) -> list:
    value = extract_obj(raw).get(key)
    return value if isinstance(value, list) else []


def minlen_ok(v):
    return isinstance(v, str) and len(v.split()) >= 8

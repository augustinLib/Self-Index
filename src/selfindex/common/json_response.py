import json
import re

_JSON = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)
_SURROGATES = re.compile("[\ud800-\udfff]")


def extract_json(text):
    if not isinstance(text, str):
        return None
    body = _strip_wrappers(text)
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = _salvage_json(body)
    return _without_surrogates(parsed)


def _without_surrogates(value):
    if isinstance(value, str):
        return _SURROGATES.sub("", value)
    if isinstance(value, list):
        return [_without_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {key: _without_surrogates(item) for key, item in value.items()}
    return value


def _strip_wrappers(text):
    body = text.strip()
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1].strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body).strip()
    return body


def _salvage_json(body):
    if match := _JSON.search(body):
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    for suffix in ('"]}', '"}]}', "]}", "}"):
        try:
            return json.loads(body + suffix)
        except Exception:
            continue
    listed = re.search(r'"views"\s*:\s*\[(.*)', body, re.S)
    if listed and (items := re.findall(r'"((?:[^"\\]|\\.)*)"', listed.group(1))):
        try:
            return {"views": [json.loads('"%s"' % item) for item in items]}
        except Exception:
            return {"views": items}
    return None

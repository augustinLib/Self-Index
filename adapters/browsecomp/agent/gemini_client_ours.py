import asyncio
import contextvars
import random
import re
from functools import partial

from browsecomp.adapters.compatibility import patched_attributes

MAX_ATTEMPTS = 12
DEFAULT_RETRY_SECONDS = 15.0
RETRY_JITTER_SECONDS = 8.0
MAX_RETRY_SECONDS = 90.0
RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))


def _safe_filter(original, schema):
    if not isinstance(schema, dict):
        return schema
    return original(schema)


_TURN_USAGE = contextvars.ContextVar("afc_usage_acc", default=None)
_USAGE_FIELDS = (
    "prompt_token_count",
    "cached_content_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "tool_use_prompt_token_count",
    "total_token_count",
)


def _is_retryable(error):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    message = str(error)
    return (
        code in RETRYABLE_STATUS
        or "RESOURCE_EXHAUSTED" in message
        or "UNAVAILABLE" in message
    )


def _retry_delay(error):
    match = re.search(r"retry in ([0-9.]+)s", str(error))
    delay = (
        float(match.group(1)) if match else DEFAULT_RETRY_SECONDS
    ) + random.uniform(0, RETRY_JITTER_SECONDS)
    return min(delay, MAX_RETRY_SECONDS)


async def _generate_with_retry(original, self, *args, **kwargs):
    response = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await original(self, *args, **kwargs)
            break
        except Exception as error:
            if not _is_retryable(error) or attempt == MAX_ATTEMPTS - 1:
                raise
            await asyncio.sleep(_retry_delay(error))
    turn_usage = _TURN_USAGE.get()
    if turn_usage is not None and getattr(response, "usage_metadata", None):
        turn_usage.append(response.usage_metadata)
    return response


async def _generate_with_summed_usage(original, self, *args, **kwargs):
    turn_usage = []
    token = _TURN_USAGE.set(turn_usage)
    try:
        response = await original(self, *args, **kwargs)
    finally:
        _TURN_USAGE.reset(token)
    if len(turn_usage) > 1 and getattr(response, "usage_metadata", None):
        for name in _USAGE_FIELDS:
            total = sum(getattr(usage, name, None) or 0 for usage in turn_usage)
            if total:
                setattr(response.usage_metadata, name, total)
    return response


def _sanitize_dump(dump):
    if not isinstance(dump, dict):
        return dump
    if not dump.get("candidates"):
        dump["candidates"] = [{"content": {"parts": []}}]
    for content in dump.get("automatic_function_calling_history") or []:
        for part in (content or {}).get("parts") or []:
            function_response = (part or {}).get("function_response")
            if isinstance(function_response, dict):
                _sanitize_function_response(function_response)
    return dump


def _sanitize_function_response(function_response):
    response = function_response.get("response")
    if not isinstance(response, dict):
        function_response["response"] = response = {}
    result = response.get("result")
    if not isinstance(result, dict):
        response["result"] = result = {}
    content = result.get("content")
    if not (isinstance(content, list) and content):
        result["content"] = [{"text": ""}]
    elif not isinstance(content[0], dict):
        result["content"] = [{"text": str(content[0])}]


class _SafeResp:

    def __init__(self, response):
        self._response = response

    def __getattr__(self, name):
        return getattr(self._response, name)

    def model_dump(self, **options):
        return _sanitize_dump(self._response.model_dump(**options))


def _persist_safe(
    original, out_dir, request_config, response, query_id=None, store_raw=False
):
    return original(
        out_dir,
        request_config,
        _SafeResp(response),
        query_id=query_id,
        store_raw=store_raw,
    )


def main():
    import mcp.client.session as sessions
    from google.genai import _mcp_utils
    from google.genai.models import AsyncModels

    from browsecomp.agent._bootstrap import upstream_module

    client = upstream_module("gemini_client")
    original_inner = AsyncModels._generate_content
    original_outer = AsyncModels.generate_content

    async def inner(self, *args, **kwargs):
        return await _generate_with_retry(original_inner, self, *args, **kwargs)

    async def outer(self, *args, **kwargs):
        return await _generate_with_summed_usage(original_outer, self, *args, **kwargs)

    def retain_session(self, _memo=None):
        return self

    with patched_attributes(
        (sessions.ClientSession, "__deepcopy__", retain_session),
        (
            _mcp_utils,
            "_filter_to_supported_schema",
            partial(_safe_filter, _mcp_utils._filter_to_supported_schema),
        ),
        (AsyncModels, "_generate_content", inner),
        (AsyncModels, "generate_content", outer),
        (client, "_persist_response", partial(_persist_safe, client._persist_response)),
    ):
        asyncio.run(client.main())

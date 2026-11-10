import os
import threading
import time

from selfindex.runtime.platform.endpoints import api_key
from browsecomp.adapters.compatibility import patched_attributes
from browsecomp.adapters.search_factory import key_set_from

AGENT_TIMEOUT_SECONDS = 600.0
_DEADLINE_START_NUMERATOR, _DEADLINE_START_DENOMINATOR = 19, 20
_DEADLINE_MEDIAN_MULTIPLE = 2
_DEADLINE_LOCK = threading.Lock()
_COMPLETION_SECONDS = []
_PENDING_QUERY_COUNT = [0]
_ABORTED_QUERIES = []
_QUERY_CLOCK = threading.local()
_QUERY_IDS_BY_TEXT = {}


class _QueryDeadline(BaseException): ...


def _deadline_elapsed():
    if (started := getattr(_QUERY_CLOCK, "started", None)) is None:
        return None
    elapsed = time.monotonic() - started
    with _DEADLINE_LOCK:
        total = _PENDING_QUERY_COUNT[0]
        n = len(_COMPLETION_SECONDS)
        if (
            total <= 0
            or n * _DEADLINE_START_DENOMINATOR < total * _DEADLINE_START_NUMERATOR
        ):
            return None
        durations = sorted(_COMPLETION_SECONDS)
        median = durations[len(durations) // 2]
    return elapsed if elapsed > _DEADLINE_MEDIAN_MULTIPLE * median else None


def _query_id(initial_request):
    try:
        content = initial_request["input"][0]["content"]
    except Exception:
        return "?"
    for text, query_id in _QUERY_IDS_BY_TEXT.items():
        if text and text in content:
            return query_id
    return "?"


class _LeastLoadedClient:

    class _Responses:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **request):
            outer = self.outer
            if (elapsed := _deadline_elapsed()) is not None:
                raise _QueryDeadline(elapsed)
            with outer.lock:
                j = min(range(len(outer.clients)), key=lambda i: outer.inflight[i])
                outer.inflight[j] += 1
            try:
                return outer.clients[j].responses.create(**request)
            finally:
                with outer.lock:
                    outer.inflight[j] -= 1

    def __init__(self, urls):
        import openai

        self.clients = [
            openai.OpenAI(
                base_url=url,
                api_key=api_key(),
                timeout=AGENT_TIMEOUT_SECONDS,
                max_retries=0,
            )
            for url in urls
        ]
        self.inflight = [0] * len(self.clients)
        self.lock = threading.Lock()
        self.responses = self._Responses(self)

    def close(self):
        for client in self.clients:
            client.close()


def _conversation_with_deadline(
    original, client, initial_request, tool_handler, max_iterations=100, verbose=False
):
    _QUERY_CLOCK.started = time.monotonic()
    try:
        out = original(client, initial_request, tool_handler, max_iterations, verbose)
    except _QueryDeadline as deadline:
        elapsed = float(deadline.args[0]) if deadline.args else 0.0
        query_id = _query_id(initial_request)
        with _DEADLINE_LOCK:
            _ABORTED_QUERIES.append((query_id, elapsed / 60.0))
            aborted = len(_ABORTED_QUERIES)
        print(
            f"[deadline] query {query_id} aborted after {elapsed / 60:.1f}min; "
            f"the next pass runs it again (aborted so far {aborted})",
            flush=True,
        )
        raise
    with _DEADLINE_LOCK:
        _COMPLETION_SECONDS.append(time.monotonic() - _QUERY_CLOCK.started)
    return out


def _parse_args():
    from browsecomp.agent.ours_agent_common import agent_parser

    parser = agent_parser()
    parser.add_argument("--model-url", required=True, help="comma-separated replicas")
    return parser.parse_args()


def _register_queries(query_path, output_dir):
    import csv
    import glob

    queries = []
    with open(query_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))
    for query_id, text in queries:
        _QUERY_IDS_BY_TEXT[text] = query_id
    saved = len(glob.glob(os.path.join(output_dir, "run_*.json")))
    _PENDING_QUERY_COUNT[0] = max(0, len(queries) - saved)
    print(
        f"[deadline] this pass {_PENDING_QUERY_COUNT[0]} queries "
        f"(after {_DEADLINE_START_NUMERATOR}/{_DEADLINE_START_DENOMINATOR} of them, "
        f"a query over {_DEADLINE_MEDIAN_MULTIPLE}x the median time is aborted)",
        flush=True,
    )


def _upstream_hooks(upstream):
    original_conversation = upstream.run_conversation_with_tools

    def conversation(*args, **kwargs):
        return _conversation_with_deadline(original_conversation, *args, **kwargs)

    return ((upstream, "run_conversation_with_tools", conversation),)


def _report_aborted():
    if _ABORTED_QUERIES:
        print(
            "[deadline] queries aborted this pass "
            + ", ".join(f"{q}({m:.0f}min)" for q, m in _ABORTED_QUERIES),
            flush=True,
        )


def main():
    from browsecomp.agent._bootstrap import upstream_module

    upstream = upstream_module("oss_client")
    with _DEADLINE_LOCK:
        _COMPLETION_SECONDS.clear()
        _ABORTED_QUERIES.clear()
        _QUERY_IDS_BY_TEXT.clear()
    options = _parse_args()
    keys = key_set_from(options)
    _register_queries(options.query, options.output_dir)

    from browsecomp.agent.ours_agent_common import (
        build_searcher,
        search_tool_handler,
        upstream_arguments,
    )
    from selfindex.llm.backbones import AgentBackbone

    searcher = build_searcher(options.retriever, keys, label="gpt-oss")
    handler = search_tool_handler(upstream, searcher)
    urls = [url.strip() for url in options.model_url.split(",") if url.strip()]
    client = _LeastLoadedClient(urls)
    print(f"[gpt-oss] endpoints {len(urls)}", flush=True)
    backbone = AgentBackbone.for_model(options.model)
    run_settings = upstream_arguments(
        options, verbose=False, reasoning_effort=backbone.reasoning_effort
    )
    hooks = _upstream_hooks(upstream)
    try:
        with patched_attributes(*hooks):
            upstream._process_tsv_dataset(options.query, client, run_settings, handler)
    finally:
        try:
            _report_aborted()
        finally:
            client.close()
    print("[gpt-oss] agent pass finished", flush=True)

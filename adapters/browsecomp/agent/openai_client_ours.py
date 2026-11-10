import hashlib
import threading

from browsecomp.adapters.search_factory import key_set_from


def _with_prompt_cache_key(build_request):

    def build(**request):
        body = build_request(**request)
        body["prompt_cache_key"] = (
            "bcp-" + hashlib.md5(request["query"].encode()).hexdigest()[:16]
        )
        return body

    return build


def main():
    import openai

    from browsecomp.adapters.compatibility import patched_attributes
    from browsecomp.agent._bootstrap import upstream_module
    from browsecomp.agent.ours_agent_common import (
        agent_parser,
        build_searcher,
        report_thread_exception,
        search_tool_handler,
        upstream_arguments,
    )
    from selfindex.llm.backbones import AgentBackbone
    from selfindex.runtime.platform.endpoints import endpoint

    upstream = upstream_module("search_agent.openai_client")
    options = agent_parser().parse_args()
    connection = endpoint("AGENT", model=options.model)
    print(f"[openai-agent] model={options.model}", flush=True)

    searcher = build_searcher(
        options.retriever, key_set_from(options), label="openai-agent"
    )
    handler = search_tool_handler(upstream, searcher)
    client = openai.OpenAI(base_url=connection.base_url, api_key=connection.api_key)
    backbone = AgentBackbone.for_model(options.model)
    run_settings = upstream_arguments(
        options,
        system=None,
        temperature=None,
        top_p=None,
        reasoning_effort=backbone.reasoning_effort,
    )
    build_request = _with_prompt_cache_key(upstream.build_request)
    with (
        client,
        patched_attributes(
            (upstream, "build_request", build_request),
            (threading, "excepthook", report_thread_exception),
        ),
    ):
        upstream._process_tsv_dataset(options.query, client, run_settings, handler)
    print("[openai-agent] agent pass finished", flush=True)

import argparse

from browsecomp.adapters.search_factory import add_search_arguments, key_set_from


def main():
    from fastmcp import FastMCP

    from browsecomp.agent._bootstrap import upstream_module
    from browsecomp.agent.ours_agent_common import SEARCH_TOOL_OPTIONS, build_searcher

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    add_search_arguments(parser)
    options = parser.parse_args()

    register_tools = upstream_module("tools").register_tools
    searcher = build_searcher(options.retriever, key_set_from(options), label="mcp")
    mcp = FastMCP("browsecomp-plus-ours")
    register_tools(mcp, searcher, **SEARCH_TOOL_OPTIONS)
    print(
        f"[mcp] http://127.0.0.1:{options.port}/sse start "
        "(SSE, as gemini_client's SSETransport expects)",
        flush=True,
    )
    mcp.run(transport="sse", host="127.0.0.1", port=options.port)

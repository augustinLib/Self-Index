import argparse
from types import SimpleNamespace

from browsecomp.adapters.search_factory import add_search_arguments, get_search_adapter
from selfindex.common.models.roles import COUNT_TOKENIZER

SEARCH_TOOL_OPTIONS = {"snippet_max_tokens": 0, "k": 5, "include_get_document": False}
QUERY_TEMPLATE = "QUERY_TEMPLATE_NO_GET_DOCUMENT"
MAX_ITERATIONS = 100


def agent_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query", required=True, help="query TSV")
    parser.add_argument("--num-threads", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    add_search_arguments(parser)
    return parser


def original_corpus():
    from selfindex.data.corpora import load_bcplus

    corpus = load_bcplus()
    return dict(zip(corpus.ids, corpus.texts, strict=True))


def build_searcher(retriever, keys, label="agent"):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COUNT_TOKENIZER)
    directory = keys.directory if keys is not None else "base index"
    print(f"[{label}] retriever={retriever} keys={directory}", flush=True)
    return get_search_adapter(retriever).create(keys, tokenizer, original_corpus())


def search_tool_handler(upstream, searcher):
    return upstream.SearchToolHandler(searcher, **SEARCH_TOOL_OPTIONS)


def upstream_arguments(options, **extra):
    return SimpleNamespace(
        model=options.model,
        max_tokens=options.max_tokens,
        output_dir=options.output_dir,
        query_template=QUERY_TEMPLATE,
        num_threads=options.num_threads,
        max_iterations=MAX_ITERATIONS,
        **extra,
    )


def agent_urls():
    from selfindex.runtime.platform.endpoints import Endpoint, endpoint

    urls = [url.strip() for url in endpoint("AGENT").base_url.split(",")]
    return [Endpoint(url).base_url for url in urls if url]


def report_thread_exception(exc):
    print(f"[thread exception] {exc.exc_type.__name__}: {exc.exc_value}", flush=True)

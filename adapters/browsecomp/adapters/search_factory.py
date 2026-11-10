import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from browsecomp.adapters.search import SearchAdapter
from browsecomp.agent._paths import BROWSECOMP_DIR, INDEXES
from selfindex.runtime.commands import COMMANDS
from selfindex.runtime.registry import Registry

KEY_ROOT = Path(BROWSECOMP_DIR) / "agent"


@dataclass(frozen=True)
class KeySet:

    directory: Path
    index: Path | None = None

    def arguments(self):
        index = ["--index", str(self.index)] if self.index is not None else []
        return ["--keys", str(self.directory), *index]


def add_search_arguments(parser):
    parser.add_argument("--retriever", required=True, choices=list(SEARCH_ADAPTERS))
    parser.add_argument("--keys", help="key-set directory (omit for the base index)")
    parser.add_argument("--index", help="joint document and key index of the key set")


def key_set_from(options):
    if not options.keys:
        if options.index:
            raise ValueError("--index needs the key set's --keys")
        return None
    return KeySet(Path(options.keys), Path(options.index) if options.index else None)


class SearchAdapterFactory(ABC):
    name = ""
    index_steps = None
    evolution_retriever = ""
    joint_index = False

    def key_set(self, tag):
        index = Path(INDEXES) / f"{self.name}_{tag}" if self.joint_index else None
        return KeySet(KEY_ROOT / tag, index)

    def workspace_key_set(self, root):
        index = Path(root) / f"{self.name}-index" if self.joint_index else None
        return KeySet(Path(root) / "keys", index)

    @abstractmethod
    def build(self, keys, tokenizer, corpus_text): ...

    def create(self, keys, tokenizer, corpus_text):
        adapter = self.build(keys, tokenizer, corpus_text)
        if not isinstance(adapter, SearchAdapter):
            raise TypeError("search factory must return SearchAdapter")
        return adapter

    def preparation_steps(self, keys=None):
        if self.index_steps is None:
            raise ValueError(
                f"{self.name}: full execution requires declared index_steps"
            )
        steps = []
        for command, argv in self.index_steps:
            if command not in COMMANDS:
                raise ValueError(f"unknown index preparation command: {command}")
            if isinstance(argv, str) or any(not isinstance(arg, str) for arg in argv):
                raise TypeError(
                    "index preparation arguments must be a sequence of strings"
                )
            steps.append((command, [*argv, *(keys.arguments() if keys else ())]))
        return steps


class _BuiltinSearchFactory(SearchAdapterFactory):
    def __init__(self, name, command, evolution_retriever, joint_index=False):
        self.name = name
        self.index_steps = ((command, ("--retriever", name) if joint_index else ()),)
        self.evolution_retriever = evolution_retriever
        self.joint_index = joint_index

    def build(self, keys, tokenizer, corpus_text):
        from browsecomp.agent.searchers_ours import BM25Searcher, DenseSearcher

        adapter = BM25Searcher if self.joint_index else DenseSearcher
        return adapter(keys, tokenizer, corpus_text)


SEARCH_ADAPTERS = Registry("BrowseComp search adapter")


def register_search_adapter(factory):
    if not isinstance(factory, SearchAdapterFactory):
        raise TypeError("factory must implement SearchAdapterFactory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", factory.name):
        raise ValueError("search adapter name must be a simple path component")
    return SEARCH_ADAPTERS.register(factory.name, factory)


def get_search_adapter(name):
    if name not in SEARCH_ADAPTERS:
        raise ValueError(
            f"unsupported BrowseComp search adapter: {name}; "
            f"available: {', '.join(SEARCH_ADAPTERS)}"
        )
    return SEARCH_ADAPTERS[name]


def search_adapter_name(retriever=None):
    if retriever is None:
        return "dense"
    name = {"bm25": "bm25", "qwen3": "dense"}.get(retriever.name, retriever.name)
    return get_search_adapter(name).name


for _name in ("bm25", "bm25-concat"):
    register_search_adapter(
        _BuiltinSearchFactory(_name, "bc.build_index", "bm25", joint_index=True)
    )
register_search_adapter(_BuiltinSearchFactory("dense", "bc.encode_keys", "qwen3"))

import argparse
import fcntl
import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from selfindex.common.io import read_lines
from selfindex.runtime.catalog.retrievers import get_backend
from browsecomp.adapters.search_factory import SEARCH_ADAPTERS
from browsecomp.agent._paths import INDEXES

BROWSECOMP_ROOT = Path(__file__).resolve().parents[1]

INDEX_THREADS = 24
STDOUT_TAIL_CHARS = 1500
STDERR_TAIL_CHARS = 2000


def _write_collection(collection, keys_path, entries):
    from selfindex.data.corpora import load_bcplus

    corpus = load_bcplus()
    with (collection / "docs_orig.jsonl").open("w", encoding="utf-8") as stream:
        for doc_id, text in zip(corpus.ids, corpus.texts, strict=True):
            output_record = {"id": doc_id, "contents": text}
            record_line = json.dumps(output_record, ensure_ascii=False) + "\n"
            stream.write(record_line)
    print(f"[bm25] original collection written {len(corpus.texts):,}", flush=True)
    keys = {}
    for line in read_lines(keys_path):
        record = json.loads(line)
        keys.setdefault(record["docid"], []).append(record["k"])
    with (collection / "docs_keys.jsonl").open("w", encoding="utf-8") as stream:
        for doc_id, texts in keys.items():
            each = enumerate(texts) if entries == "all" else ()
            records = [(f"{doc_id}#k{i}", text) for i, text in each]
            records.append((f"{doc_id}#k", "\n".join(texts)))
            for entry_id, contents in records:
                output_record = {"id": entry_id, "contents": contents}
                record_line = json.dumps(output_record, ensure_ascii=False) + "\n"
                stream.write(record_line)
    print(f"[bm25] {entries} key entries of {len(keys):,} documents", flush=True)


@contextmanager
def _publication_lock(destination):
    lock = destination.with_name(f".{destination.name}.build.lock")
    with lock.open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise FileExistsError(
                f"another build is publishing {destination}"
            ) from None
        yield


def _publish_index(staged, destination):
    if not staged.is_dir() or not any(staged.iterdir()):
        raise ValueError(f"index builder produced no index: {staged}")
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.publish-", dir=destination.parent
    ) as pending:
        candidate = Path(pending) / "index"
        if staged.stat().st_dev == destination.parent.stat().st_dev:
            staged.rename(candidate)
        else:
            shutil.copytree(staged, candidate)
        backup = None
        if destination.exists():
            backup = (
                Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.previous-", dir=destination.parent
                    )
                )
                / "index"
            )
            destination.rename(backup)
        try:
            candidate.rename(destination)
        except OSError:
            if backup is not None:
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup.parent)


def _build_index(collection, index):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection),
            "--index",
            str(index),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(INDEX_THREADS),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-STDOUT_TAIL_CHARS:], flush=True)
    if result.returncode:
        print(result.stderr[-STDERR_TAIL_CHARS:], flush=True)
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _index_destination(index):
    destination = Path(INDEXES) / index
    if destination.is_symlink():
        raise ValueError("index destination must not be a symlink")
    destination = destination.resolve()
    if (
        destination == Path(INDEXES).resolve()
        or destination in BROWSECOMP_ROOT.resolve().parents
        or destination == BROWSECOMP_ROOT
    ):
        raise ValueError("index destination must be a dedicated index directory")
    if destination.exists() and not destination.is_dir():
        raise ValueError("index destination must be a directory")
    return destination


def _temporary_roots(keys_dir, destination):
    collection_root = keys_dir / "bm25_coll"
    staging_root = destination.parent
    if collection_root.resolve().is_relative_to(destination):
        raise ValueError(
            "temporary build directories must be outside the published index"
        )
    return collection_root, staging_root


def _parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        required=True,
        choices=[
            name for name, adapter in SEARCH_ADAPTERS.items() if adapter.joint_index
        ],
        help="BM25 retriever whose document scores the index serves",
    )
    parser.add_argument(
        "--keys",
        required=True,
        help="key-set directory (relative: under adapters/browsecomp/)",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="index directory to publish "
        "(relative: under adapters/browsecomp/indexes/)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = _parse_args(argv)
    entries = get_backend(options.retriever).entries
    keys_dir = BROWSECOMP_ROOT / options.keys
    destination = _index_destination(options.index)
    collection_root, staging_root = _temporary_roots(keys_dir, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    collection_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    with (
        _publication_lock(destination),
        tempfile.TemporaryDirectory(
            dir=collection_root, prefix="collection-"
        ) as collection,
        tempfile.TemporaryDirectory(dir=staging_root, prefix="build-") as staging,
    ):
        _write_collection(Path(collection), keys_dir / "keys.jsonl", entries)
        index = Path(staging) / "index"
        _build_index(Path(collection), index)
        _publish_index(index, destination)
    print(f"[bm25] index published at {destination}", flush=True)
    return 0

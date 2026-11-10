import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def write_json(self, name, value, **options): ...

    @abstractmethod
    def write_jsonl(self, name, records): ...


_MADE_DIRECTORIES = set()


def _make_directory(directory, again=False):
    if again or directory not in _MADE_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
        _MADE_DIRECTORIES.add(directory)


def atomic_replace(path, write, **temporary_options):
    path = Path(path)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, **temporary_options
        ) as stream:
            pending = Path(stream.name)
            write(stream)
        os.replace(pending, path)
        pending = None
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root):
        self.root = Path(root)

    def _atomic(self, name, write):
        path = self.root / name
        _make_directory(path.parent)
        options = dict(
            mode="w",
            encoding="utf-8",
            errors="backslashreplace",
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            atomic_replace(path, write, **options)
        except FileNotFoundError:
            _make_directory(path.parent, again=True)
            atomic_replace(path, write, **options)

    def write_json(self, name, value, **options):
        self._atomic(name, lambda fh: json.dump(value, fh, **options))

    def write_jsonl(self, name, records):
        def write(fh):
            for record in records:
                record_line = json.dumps(record, ensure_ascii=False) + "\n"
                fh.write(record_line)

        self._atomic(name, write)

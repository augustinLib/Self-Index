import hashlib
import json
from pathlib import Path

from selfindex.pipeline.settings import PARAMETERS
from selfindex.storage.artifacts import LocalArtifactStore

CONFIGURATION_VARIABLES = frozenset(PARAMETERS) | frozenset(
    f"{role}_BASE_URL" for role in ("LLM", "EMBEDDING", "AGENT", "READER", "JUDGE")
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_files(root, packages):
    return [
        path for name in packages for path in sorted((Path(root) / name).rglob("*.py"))
    ]


def source_digests(root, packages):
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in source_files(root, packages)
    }


def configuration_environment(env):
    return {key: value for key, value in env.items() if key in CONFIGURATION_VARIABLES}


class WorkflowCheckpoint:
    def __init__(self, root, identity, names, resume):
        self.root = Path(root).resolve()
        self.store = LocalArtifactStore(self.root)
        self.path = self.root / "workflow_receipts.json"
        self.record = {"identity": identity, "stages": names, "receipts": {}}
        if self.path.exists():
            if not resume:
                raise FileExistsError("workflow receipts exist; enable explicit resume")
            previous = json.loads(self.path.read_text())
            if previous.get("identity") != identity:
                raise ValueError("workflow identity changed; use a fresh experiment")
            recorded = previous.get("stages", [])
            previous["stages"] = [*recorded, *(n for n in names if n not in recorded)]
            self.record = previous
            for receipt in previous["receipts"].values():
                self.verify(receipt)
        elif resume:
            raise ValueError("resume requires an existing workflow identity")
        else:
            self.store.write_json(self.path.name, self.record)

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("workflow artifact escapes output directory")
        return path

    def verify(self, receipt):
        if not receipt:
            raise ValueError("empty workflow artifact receipt")
        for name, digest in receipt.items():
            path = self._path(name)
            if not path.is_file() or file_digest(path) != digest:
                raise ValueError(f"workflow artifact missing or changed: {name}")

    def complete(self, stage):
        return stage in self.record["receipts"]

    def commit(self, stage, paths):
        receipt = {}
        for path in paths:
            name = str(Path(path).resolve().relative_to(self.root))
            receipt[name] = file_digest(self._path(name))
        if not receipt:
            raise ValueError(f"stage produced no verifiable artifacts: {stage}")
        self.record["receipts"][stage] = receipt
        self.store.write_json(self.path.name, self.record)

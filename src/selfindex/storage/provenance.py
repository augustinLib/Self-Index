import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from selfindex.runtime.platform.environment import REPO_ROOT
from selfindex.storage.artifacts import LocalArtifactStore

SOURCE_PACKAGES = ("src/selfindex", "adapters/browsecomp", "adapters/longmem")
SOURCE_CONFIGURATION = ("requirements.txt", "pyproject.toml")
RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "torch",
    "transformers",
    "tokenizers",
    "datasets",
    "openai",
    "vllm",
    "pyserini",
    "gensim",
    "scipy",
)
GIT_TIMEOUT_SECONDS = 5
EXCLUDED_DIRECTORIES = {"runs", "logs", "results"}


def in_data_directory(root, relative):
    root, parts = Path(root), relative.parts
    for end, part in enumerate(parts):
        parent = root.joinpath(*parts[:end])
        if part == "data" and not (
            (parent / "__init__.py").is_file()
            and (parent / part / "__init__.py").is_file()
        ):
            return True
    return False


class ProvenanceWriter:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else Path(REPO_ROOT)

    def sources(self):
        for name in ("run.py", *SOURCE_CONFIGURATION):
            entry = self.root / name
            if entry.is_file() and not entry.is_symlink():
                yield entry
        for package in SOURCE_PACKAGES:
            for source in sorted((self.root / package).rglob("*.py")):
                relative = source.relative_to(self.root)
                if (
                    not EXCLUDED_DIRECTORIES.intersection(relative.parts)
                    and not in_data_directory(self.root, relative)
                    and not source.is_symlink()
                ):
                    yield source

    def _git(self, *args):
        if not (self.root / ".git").exists():
            return None
        try:
            result = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(self.root), *args],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def _revision(self):
        status = self._git("status", "--porcelain", "--untracked-files=normal")
        return {
            "head": self._git("rev-parse", "HEAD"),
            "dirty": status != "" if status is not None else None,
        }

    def _runtime(self):
        versions = {}
        for name in RUNTIME_DISTRIBUTIONS:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        return {"python": platform.python_version(), "packages": versions}

    def write(self, out):
        destination = Path(out) / "src"
        hashes = {}
        for source in self.sources():
            relative = source.relative_to(self.root)
            contents = source.read_bytes()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
            hashes[relative.as_posix()] = hashlib.sha256(contents).hexdigest()
        store = LocalArtifactStore(destination)
        record = {
            "schema_version": 2,
            "sha256": hashes,
            "git": self._revision(),
            "runtime": self._runtime(),
        }
        store.write_json("manifest.json", record, indent=2)

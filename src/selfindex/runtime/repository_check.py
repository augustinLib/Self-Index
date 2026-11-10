import argparse
import subprocess
from pathlib import Path

from selfindex.runtime.platform.environment import REPO_ROOT

REQUIRED_FILES = (
    "requirements.txt",
    "pyproject.toml",
    "setup.sh",
    "README.md",
    "run.py",
)
GENERATED_SUFFIXES = {".pyc", ".safetensors", ".npy", ".npz", ".log"}
GITLINK_MODE = "160000"


def has_git_metadata(root):
    return (root / ".git").exists()


def git_index(root):
    if not has_git_metadata(root):
        return
    output = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root)
    for entry in output.decode().split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            mode, object_id, stage = metadata.split()
            yield mode, object_id, stage, path
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root
    )
    for path in untracked.decode().split("\0"):
        if path:
            yield "100644", "", "0", path


def check_source(root, name):
    path = root / name
    if not path.exists() and not path.is_symlink():
        return []
    if path.is_symlink():
        try:
            destination = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return [f"broken or cyclic tracked symlink: {name}"]
        return (
            []
            if destination.is_relative_to(root)
            else [f"tracked symlink escapes repository: {name}"]
        )
    errors = []
    if path.suffix in GENERATED_SUFFIXES:
        errors.append(f"generated artifact tracked as source: {name}")
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeError:
        return errors
    if str(root) in contents:
        errors.append(f"checkout-specific absolute path: {name}")
    return errors


def check_repository(root):
    root = Path(root).resolve()
    errors = []
    for mode, _, stage, name in git_index(root):
        if stage != "0":
            errors.append(f"unresolved Git conflict: {name}")
        if mode == GITLINK_MODE:
            continue
        errors.extend(check_source(root, name))
    errors.extend(missing_file_errors(root))
    return errors


def missing_file_errors(root):
    return [
        f"required file missing: {name}"
        for name in REQUIRED_FILES
        if not (root / name).is_file()
    ]


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    root = Path(REPO_ROOT)
    if not has_git_metadata(root):
        print("NOTE  no Git metadata: only the required files are checked")
    errors = check_repository(root)
    for error in errors:
        print(f"FAIL  {error}")
    print("preflight: FAIL" if errors else "preflight: PASS")
    return int(bool(errors))

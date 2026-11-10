from pathlib import Path

from browsecomp.adapters.search_factory import get_search_adapter

BROWSECOMP_ROOT = Path(__file__).resolve().parents[1]
BASE_INDEX_RUNS = "base"


def run_root():
    return BROWSECOMP_ROOT / "runs"


def evaluation_root():
    return BROWSECOMP_ROOT / "evals"


def evaluation_directory(input_dir, output_dir=None):
    source = Path(input_dir)
    if not source.is_absolute():
        source = BROWSECOMP_ROOT / source
    source = source.resolve()
    root = Path(output_dir) if output_dir is not None else evaluation_root()
    try:
        relative = source.relative_to(run_root().resolve())
    except ValueError:
        parts = source.parts
        relative = Path(
            *(parts[parts.index("runs") + 1 :] if "runs" in parts else parts[1:][-4:])
        )
    return root / relative


def prepare_evaluation_directory(input_dir, output_dir):
    destination = evaluation_directory(input_dir, output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def backbone_run_dir(backbone, retriever, keys):
    name = f"{backbone}-{get_search_adapter(retriever).name}"
    if keys is None:
        return str(run_root() / BASE_INDEX_RUNS / name)
    tag = Path(keys.directory).name
    if tag in ("", ".", "..", BASE_INDEX_RUNS):
        raise ValueError("key directory must identify an experiment other than base")
    return str(run_root() / tag / name)

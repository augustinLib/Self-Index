import importlib.util
import json
from pathlib import Path

from longmem.adapter.harness_options import MEMORY_TEMPLATES, TIER
from longmem.adapter.paths import DATA_ROOT, UPSTREAM_ROOT
from selfindex.runtime.platform.endpoints import endpoint
from selfindex.storage.artifacts import LocalArtifactStore

SERVICE_PARAMETERS = (
    ("embedding_params", "EMBEDDING"),
    ("controller_params", "READER"),
)


def selfindex_memory_type(template):
    if template == "agentrunbook_r":
        return "agentrunbook_r_selfindex"
    return "rag_selfindex"


def _bind_services(params):
    connections = {key: endpoint(role) for key, role in SERVICE_PARAMETERS}
    for key, connection in connections.items():
        params[key].update(
            model=connection.model,
            base_url=connection.base_url,
            api_key_env=connection.api_key_env,
            api_key_file=None,
        )


class LongMemInputs:
    def __init__(self):
        self.upstream = UPSTREAM_ROOT.resolve()
        self.data_root = DATA_ROOT.resolve()

    def _data_module(self):
        path = self.upstream / "data/public_data.py"
        spec = importlib.util.spec_from_file_location("_selfindex_longmem_data", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _materialize(self, data, domain, runtime):
        questions = data.materialize_runtime_questions(
            data_root=self.data_root,
            domain=domain,
            question_ids=None,
            limit=None,
            output_path=runtime / "questions.json",
        )
        data.materialize_runtime_haystack(
            data_root=self.data_root,
            tier=TIER,
            selected_questions=questions,
            output_path=runtime / "haystack.json",
        )

    def prepare(self, domain, out, keys, memory):
        if memory not in MEMORY_TEMPLATES:
            raise ValueError(f"unsupported SelfIndex memory template: {memory}")
        out = Path(out).resolve()
        config = json.loads(
            (self.upstream / f"evaluation/memory_configs/{memory}.json").read_text()
        )
        config["memory_type"] = selfindex_memory_type(memory)
        params = config["memory_params"]
        params.update(
            selfindex_keys_path=str(Path(keys).resolve()),
            workspace_dir=str(out / "memory_workspace"),
            trajectories_root_dir=str(self.data_root),
        )
        _bind_services(params)
        runtime = out / "runtime_inputs"
        self._materialize(self._data_module(), domain, runtime)
        LocalArtifactStore(runtime).write_json("memory_config.json", config, indent=1)

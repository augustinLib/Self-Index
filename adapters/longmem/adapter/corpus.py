import json
from types import SimpleNamespace

from longmem.adapter.harness_options import MEMORY_TEMPLATES, TIER
from longmem.adapter.paths import DATA_ROOT, UPSTREAM_ROOT
from selfindex.common.io import read_json, read_lines
from selfindex.data.corpora import Corpus


def _shared_parameters(group, names, what):
    configurations = UPSTREAM_ROOT / "evaluation" / "memory_configs"
    values = {
        tuple(
            read_json(configurations / f"{template}.json")["memory_params"][group][name]
            for name in names
        )
        for template in MEMORY_TEMPLATES
    }
    if len(values) != 1:
        raise ValueError(f"the memory systems {what} differently")
    return values.pop()


def slice_radius():
    (radius,) = _shared_parameters(
        "index_params", ("raw_state_slice_radius",), "build raw-state entries"
    )
    return radius


def memory_encoder_settings():
    max_length, instruction = _shared_parameters(
        "embedding_params", ("max_input_tokens", "query_instruction"), "embed entries"
    )
    return {
        "max_length": int(max_length),
        "query_prefix": f"Instruct: {str(instruction).strip()}\nQuery:",
    }


def haystack_trajectories():
    haystacks = read_json(DATA_ROOT / "haystacks" / f"lme_v2_{TIER}.json")
    return {trajectory for ids in haystacks.values() for trajectory in ids}


def _simplified(trajectory):
    states = [
        {
            "state_index": index,
            "url": state.get("url"),
            "action": state.get("action"),
            "text": state.get("accessibility_tree", state.get("text")),
        }
        for index, state in enumerate(trajectory["states"])
    ]
    return {"id": trajectory["id"], "goal": trajectory.get("goal"), "states": states}


def raw_state_entries(trajectory, radius):
    import longmem.adapter._bootstrap  # noqa: F401
    import memory_modules.memory  # noqa: F401
    from memory_modules.agentrunbook_r import AgentRunbookR

    builder = SimpleNamespace(raw_state_slice_radius=radius)
    return AgentRunbookR._build_raw_state_entries(builder, _simplified(trajectory))


def entry_text(entry):
    return (
        f"Goal: {entry['goal']}\n\n"
        f"Actions in this slice:\n{entry['slice_action_sequence']}\n\n"
        f"Page states:\n{entry['slice_axtree_text']}"
    )


def memory_corpus(name):
    wanted = haystack_trajectories()
    radius = slice_radius()
    entries, found = [], set()
    for line in read_lines(DATA_ROOT / "trajectories.jsonl"):
        trajectory = json.loads(line)
        if trajectory["id"] in wanted:
            found.add(trajectory["id"])
            entries.extend(raw_state_entries(trajectory, radius))
    if missing := wanted - found:
        raise ValueError(
            f"trajectories.jsonl lacks {len(missing)} haystack trajectories"
        )
    entries.sort(
        key=lambda entry: (entry["trajectory_id"], entry["center_state_index"])
    )
    ids = [entry["entry_id"] for entry in entries]
    return Corpus(
        name=name,
        ids=ids,
        texts=[entry_text(entry) for entry in entries],
        id2idx={entry: row for row, entry in enumerate(ids)},
        queries=[],
        encoder_settings=memory_encoder_settings(),
    )

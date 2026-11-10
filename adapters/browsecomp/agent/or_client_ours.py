import contextlib
import hashlib
import json
import os
import threading
from copy import deepcopy
from dataclasses import dataclass, field

from browsecomp.adapters.compatibility import patched_attributes
from browsecomp.adapters.search_factory import key_set_from
from selfindex.common.io import read_json
from selfindex.storage.artifacts import LocalArtifactStore

API_KEY_ENV = "OPENROUTER_API_KEY"
REQUEST_TIMEOUT_SECONDS = 900.0
REQUEST_RETRIES = 10


class _Routed:

    def __init__(self, client, extra_body):
        self._client, self._extra_body = client, extra_body

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **request):
        request.setdefault("extra_body", self._extra_body)
        return self._client.chat.completions.create(**request)


DROPPED = "[dropped: context limit reached]"


def _is_context_overflow(err):
    message = str(err)
    return "context_length_exceeded" in message or "max_prompt_tokens" in message


def _shed_oldest_tool_output(messages, count):
    n = 0
    for message in messages:
        if n >= count:
            break
        if message.get("role") == "tool" and message.get("content") != DROPPED:
            message["content"] = DROPPED
            n += 1
    return n


def _empty_usage():
    return dict.fromkeys(
        (
            "prompt_tokens",
            "prompt_tokens_cached",
            "completion_tokens",
            "total_tokens",
            "reasoning_tokens",
        ),
        0,
    )


@dataclass
class ConversationState:
    messages: list = field(default_factory=list)
    normalized: list = field(default_factory=list)
    usage: dict = field(default_factory=_empty_usage)
    finish_reason: str | None = None

    def payload(self):
        return {
            "messages": self.messages,
            "normalized": self.normalized,
            "usage": self.usage,
        }

    def account(self, completion):
        if (usage := getattr(completion, "usage", None)) is None:
            return
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage[name] += getattr(usage, name, 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if (cached := getattr(prompt_details, "cached_tokens", None)) is None:
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        self.usage["prompt_tokens_cached"] += int(cached)


@dataclass
class Conversation:

    client: object
    model: str
    tool_handler: object
    ckpt_dir: str
    max_tokens: int
    query_formatter: object
    query_template: object
    max_iterations: int

    def ckpt_path(self, query):
        return os.path.join(
            self.ckpt_dir, hashlib.md5(query.encode()).hexdigest() + ".json"
        )

    def _initial_state(self, query, checkpoint):
        state = ConversationState()
        if os.path.exists(checkpoint):
            try:
                saved = read_json(checkpoint)
                state = ConversationState(
                    saved["messages"], saved["normalized"], saved["usage"]
                )
                print(
                    f"[ckpt] resume: turns {len(state.messages)} · consumed output {state.usage['completion_tokens']:,}",
                    flush=True,
                )
            except (OSError, ValueError, KeyError, TypeError):
                state = ConversationState()
        if not state.messages:
            content = self.query_formatter(query, self.query_template)
            state.messages.append({"role": "user", "content": content})
        return state

    def _completion(self, state, tools, remaining):
        options = {
            "model": self.model,
            "messages": state.messages,
            "tools": tools,
            "max_tokens": remaining,
        }
        shed = 1
        while True:
            try:
                return self.client.chat.completions.create(**options)
            except Exception as error:
                if (
                    getattr(error, "status_code", None) != 400
                    or not _is_context_overflow(error)
                    or not _shed_oldest_tool_output(state.messages, shed)
                ):
                    raise
                print(
                    f"[context] shed {shed} oldest tool outputs, retrying", flush=True
                )
                shed *= 2

    def _execute_tools(self, state, calls):
        for call in calls:
            name = call.function.name
            record = {
                "type": "tool_call",
                "tool_name": name,
                "arguments": call.function.arguments,
                "output": None,
            }
            state.normalized.append(record)
            try:
                output = self.tool_handler.execute_tool(
                    name, json.loads(call.function.arguments)
                )
            except Exception as error:
                output = f"Error executing {name}: {error}"
                print(output)
            record["output"] = output
            state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": name,
                    "content": output,
                }
            )

    def _record_completion(self, state, completion):
        choice = completion.choices[0]
        state.finish_reason = choice.finish_reason
        state.account(completion)
        message = choice.message.model_dump()
        if "reasoning_content" in message:
            state.normalized.append(
                {
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": message.pop("reasoning_content"),
                }
            )
        state.messages.append(message)
        if message.get("content") and message["content"].strip():
            state.normalized.append(
                {
                    "type": "output_text",
                    "tool_name": None,
                    "arguments": None,
                    "output": message["content"],
                }
            )
        if state.finish_reason == "tool_calls":
            self._execute_tools(state, choice.message.tool_calls)

    def run(self, *, query):
        checkpoint = self.ckpt_path(query)
        state = self._initial_state(query, checkpoint)
        store = LocalArtifactStore(self.ckpt_dir)
        tools = self.tool_handler.get_tool_definitions()
        for _ in range(self.max_iterations):
            remaining = self.max_tokens - state.usage["completion_tokens"]
            if remaining <= 0:
                print(
                    f"Warning: Reached global max_tokens output budget ({self.max_tokens})"
                )
                break
            self._record_completion(state, self._completion(state, tools, remaining))
            store.write_json(os.path.basename(checkpoint), state.payload())
            if state.finish_reason != "tool_calls":
                break
        if state.finish_reason is None:
            print(
                f"Warning: Conversation hit max iterations ({self.max_iterations}) without final response"
            )
        usage = dict(state.usage)
        reasoning = int(usage.get("reasoning_tokens") or 0)
        if reasoning and int(usage.get("completion_tokens") or 0) >= reasoning:
            usage["completion_tokens"] -= reasoning
        with contextlib.suppress(OSError):
            os.remove(checkpoint)
        return state.normalized, usage, state.finish_reason


def upstream_hook(ckpt_dir, query_formatter):

    def hook(
        client_,
        *,
        query,
        model,
        max_tokens,
        tool_handler,
        query_template,
        max_iterations,
        **unused,
    ):
        return Conversation(
            client=client_,
            model=model,
            tool_handler=tool_handler,
            ckpt_dir=ckpt_dir,
            max_tokens=max_tokens,
            query_formatter=query_formatter,
            query_template=query_template,
            max_iterations=max_iterations,
        ).run(query=query)

    return hook


def main():
    import openai

    from browsecomp.agent._bootstrap import upstream_module
    from browsecomp.agent.ours_agent_common import (
        agent_parser,
        build_searcher,
        report_thread_exception,
        search_tool_handler,
        upstream_arguments,
    )
    from selfindex.llm.backbones import AgentBackbone
    from selfindex.runtime.platform.endpoints import endpoint

    upstream = upstream_module("search_agent.glm_zai_client")
    options = agent_parser().parse_args()
    if not (key := os.environ.get(API_KEY_ENV)):
        raise SystemExit(f"{API_KEY_ENV} is not set")
    connection = endpoint("AGENT", model=options.model)
    print(f"[or-agent] model={options.model} base={connection.base_url}", flush=True)

    searcher = build_searcher(
        options.retriever, key_set_from(options), label="or-agent"
    )
    handler = search_tool_handler(upstream, searcher)
    raw = openai.OpenAI(
        api_key=key,
        base_url=connection.base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=REQUEST_RETRIES,
    )
    backbone = AgentBackbone.for_model(options.model)
    client = _Routed(raw, deepcopy(backbone.extra_body))
    run_settings = upstream_arguments(
        options, system=None, temperature=None, top_p=None
    )
    ckpt_dir = os.path.join(options.output_dir, ".ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with (
        raw,
        patched_attributes(
            (
                upstream,
                "run_conversation_with_tools",
                upstream_hook(ckpt_dir, upstream.format_query),
            ),
            (threading, "excepthook", report_thread_exception),
        ),
    ):
        upstream._process_tsv_dataset(options.query, client, run_settings, handler)
    print("[or-agent] agent pass finished", flush=True)

import ast
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from browsecomp.agent._paths import UPSTREAM
from selfindex.llm.backbones import JudgeBackbone

SDK_FIELDS = ("model", "messages", "temperature", "top_p", "max_tokens")


@dataclass(frozen=True)
class JudgeSampling:
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int


class AnswerJudge(ABC):
    @abstractmethod
    def chat(self, conversations, sampling, *, chat_template_kwargs): ...

    def close(self):  # noqa: B027
        ...


class RemoteAnswerJudge(AnswerJudge):

    def __init__(self, connection):
        self.client = connection.client()
        self.backbone = JudgeBackbone.for_model(connection.model)

    def chat(self, conversations, sampling, *, chat_template_kwargs):
        conversations = list(conversations)
        if not conversations:
            return []
        with ThreadPoolExecutor(max_workers=len(conversations)) as pool:
            texts = list(
                pool.map(
                    lambda messages: self._complete(
                        messages, sampling, chat_template_kwargs
                    ),
                    conversations,
                )
            )
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in texts]

    def _complete(self, messages, sampling, chat_template_kwargs):
        body = self.backbone.body(
            messages,
            sampling.temperature,
            sampling.max_tokens,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            chat_template_kwargs=dict(chat_template_kwargs),
        )
        options = {name: body.pop(name) for name in SDK_FIELDS if name in body}
        response = self.client.chat.completions.create(**options, extra_body=body)
        text = self.backbone.text(response.choices[0].message.model_dump())
        if not isinstance(text, str):
            raise ValueError("judge response has no textual answer")
        return text

    def close(self):
        self.client.close()


def load_evaluator(engine_factory, sampling_factory):
    source = Path(UPSTREAM) / "scripts_evaluation/evaluate_run.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "vllm"
    ]
    if len(imports) != 1 or {alias.name for alias in imports[0].names} != {
        "LLM",
        "SamplingParams",
    }:
        raise RuntimeError(
            "upstream evaluator inference contract changed; review the adapter"
        )
    tree.body.remove(imports[0])
    namespace = {
        "__name__": "_selfindex_evaluator",
        "__file__": str(source),
        "LLM": engine_factory,
        "SamplingParams": sampling_factory,
    }
    original_path = list(sys.path)
    try:
        exec(compile(tree, str(source), "exec"), namespace)
    finally:
        sys.path[:] = original_path
    return namespace

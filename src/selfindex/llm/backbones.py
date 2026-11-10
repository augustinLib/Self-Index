from copy import deepcopy

from selfindex.common.models.roles import KIMI_AGENT_MODEL, LME_EVALUATOR


class Backbone:

    reasoning_effort = None
    extra_body = {}

    def __init__(self, model):
        self.model = model

    @classmethod
    def register(cls, *models):
        def register(backbone):
            cls.backbones.update(dict.fromkeys(models, backbone))
            return backbone

        return register

    @classmethod
    def for_model(cls, model):
        return cls.backbones.get(model, cls)(model)

    def body(self, messages, temperature, max_tokens, **fields):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **fields,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return {**body, **deepcopy(self.extra_body)}

    def text(self, message):
        content = message["content"]
        return content if isinstance(content, str) else None


class OptimizerBackbone(Backbone):

    backbones = {}
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}


class AgentBackbone(Backbone):

    backbones = {}
    reasoning_effort = "high"


class JudgeBackbone(Backbone):

    backbones = {}


@AgentBackbone.register(KIMI_AGENT_MODEL)
class KimiK25Agent(AgentBackbone):

    extra_body = {
        "provider": {"order": ["SiliconFlow", "DeepInfra"], "allow_fallbacks": False}
    }


@JudgeBackbone.register(LME_EVALUATOR)
class Gpt52Judge(JudgeBackbone):

    reasoning_effort = "medium"

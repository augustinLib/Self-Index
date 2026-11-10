import json
import os
import urllib.request
from dataclasses import dataclass
from functools import cache
from urllib.parse import urlsplit

API_KEY_ENV = "OPENAI_API_KEY"
SERVICE_TIMEOUT_SECONDS = 300.0
DISCOVERY_TIMEOUT_SECONDS = 30


def api_key():
    return os.environ.get(API_KEY_ENV) or "EMPTY"


@cache
def served_model_entry(base_url):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key()}"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=DISCOVERY_TIMEOUT_SECONDS
        ) as response:
            models = json.load(response).get("data") or []
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot list the models served at {base_url}: {error}")
    if not models or not isinstance(models[0].get("id"), str):
        raise ValueError(f"{base_url} serves no model")
    return models[0]


def served_model(base_url):
    return served_model_entry(base_url)["id"]


@dataclass(frozen=True)
class Endpoint:

    base_url: str
    model_name: str = ""

    api_key_env = API_KEY_ENV

    def __post_init__(self):
        if not isinstance(self.base_url, str) or not isinstance(self.model_name, str):
            raise ValueError("endpoint URL and model must be strings")
        _validate_url(self.base_url)
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))

    @property
    def model(self):
        return self.model_name.strip() or served_model(self.base_url)

    @property
    def api_key(self):
        return api_key()

    def client(self):
        from openai import OpenAI

        return OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=SERVICE_TIMEOUT_SECONDS,
            max_retries=0,
        )


def _validate_url(base_url):
    url = urlsplit(base_url)
    if url.scheme not in ("http", "https") or not url.netloc:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError(
            "endpoint URL must not contain credentials or query parameters"
        )


def endpoint(role, *, model=""):
    if not (url := os.environ.get(f"{role}_BASE_URL", "")):
        raise ValueError(f"configure {role}_BASE_URL")
    return Endpoint(url, model)


def llm_urls():
    urls = os.environ.get("LLM_BASE_URL", "")
    return [url.strip().rstrip("/") for url in urls.split(",") if url.strip()]

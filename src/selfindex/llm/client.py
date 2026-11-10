from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import httpx
from httpx import create_ssl_context

from selfindex.llm.backbones import OptimizerBackbone
from selfindex.llm.requests import (
    REQUEST_ATTEMPTS,
    REQUEST_TIMEOUT_SECONDS,
    RequestRejected,
    permanent_client_error,
    retry_delay,
)
from selfindex.runtime.platform.endpoints import (
    Endpoint,
    api_key,
    llm_urls,
    served_model,
)

REQUESTS_PER_REPLICA = 768


class _Sender:

    def __init__(self, connect, urls, usage, text):
        self.connect, self.urls, self.usage, self.text = connect, urls, usage, text
        self._lock = threading.Lock()
        self._local = threading.local()
        self._clients = []

    def _client(self):
        if (client := getattr(self._local, "client", None)) is None:
            client = self._local.client = self.connect()
            with self._lock:
                self._clients.append(client)
        return client

    def close(self):
        for client in self._clients:
            client.close()

    def request(self, index, body):
        for attempt in range(REQUEST_ATTEMPTS):
            try:
                return self._post(self.urls[(index + attempt) % len(self.urls)], body)
            except RequestRejected:
                raise
            except Exception:
                if attempt < REQUEST_ATTEMPTS - 1:
                    time.sleep(retry_delay(attempt))
        return None

    def _post(self, url, body):
        response = self._client().post(url, content=json.dumps(body))
        if permanent_client_error(response.status_code):
            raise RequestRejected(
                f"LLM request rejected (HTTP {response.status_code}): "
                f"{response.text[:500]}"
            )
        if response.status_code != 200:
            raise RuntimeError(f"http {response.status_code}")
        payload = response.json()
        self._record_usage(payload)
        if not (choices := payload.get("choices") or []):
            raise RuntimeError("empty choices")
        text = self.text(choices[0]["message"])
        return text if isinstance(text, str) else None

    def _record_usage(self, payload):
        if not (usage := payload.get("usage") or {}):
            return
        details = usage.get("prompt_tokens_details") or {}
        counts = {
            "calls": 1,
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "cached": int(details.get("cached_tokens") or 0),
        }
        with self._lock:
            for name, value in counts.items():
                self.usage[name] = self.usage.get(name, 0) + value


def send(bodies, urls, *, requests_per_replica, api_key, usage, text):
    if not bodies:
        return []
    endpoints = [url.rstrip("/") + "/chat/completions" for url in urls]
    workers = min(requests_per_replica * len(endpoints), len(bodies))
    connect = partial(
        httpx.Client,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=create_ssl_context(),
    )
    sender = _Sender(connect, endpoints, usage, text)
    try:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(sender.request, range(len(bodies)), bodies))
    finally:
        sender.close()


def _clean_text(text):
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", "replace").decode("utf-8")


def _clean_messages(messages):
    return [
        (
            dict(message, content=_clean_text(message["content"]))
            if isinstance(message, dict) and isinstance(message.get("content"), str)
            else message
        )
        for message in messages
    ]


class LLM:

    def __init__(self):
        self.urls = [Endpoint(url).base_url for url in llm_urls()]
        if not self.urls:
            raise ValueError(
                "No LLM endpoint configured; set LLM_BASE_URL or --llm-url"
            )
        self.model = served_model(self.urls[0])
        self.backbone = OptimizerBackbone.for_model(self.model)
        self.api_key = api_key()
        self._lock = threading.Lock()
        self.tok = {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}

    def _add_usage(self, counts):
        with self._lock:
            for key, value in counts.items():
                self.tok[key] = self.tok.get(key, 0) + int(value)

    def batch(self, list_messages, temperature=0.4, max_tokens=700, per_job=None):
        list_messages = [_clean_messages(m) for m in list_messages]
        bodies = []
        for i, messages in enumerate(list_messages):
            job = per_job[i] if per_job else {}
            bodies.append(
                self.backbone.body(
                    messages, temperature, job.get("max_tokens", max_tokens)
                )
            )
        usage = {}
        try:
            replies = send(
                bodies,
                self.urls,
                requests_per_replica=REQUESTS_PER_REPLICA,
                api_key=self.api_key,
                usage=usage,
                text=self.backbone.text,
            )
        finally:
            self._add_usage(usage)
        return replies

import os
import shlex
import socket
import subprocess
import time
from abc import ABC, abstractmethod

from selfindex.common.io import read_json, write_json
from selfindex.common.models.roles import (
    AGENT_MODEL,
    GEMINI_AGENT_MODEL,
    KIMI_AGENT_MODEL,
    NANO_AGENT_MODEL,
)
from browsecomp.agent import or_client_ours
from browsecomp.costs.usage import PRICES, usage_summary
from selfindex.runtime.commands import subprocess_argv
from selfindex.runtime.platform.endpoints import API_KEY_ENV
from browsecomp.agent._paths import BROWSECOMP_DIR


def base_env():
    from selfindex.runtime.platform.environment import ensure_env

    env = dict(os.environ)
    env["HF_HOME"] = ensure_env().hf_home
    return env


_REGISTRY = {}


def agent_names():
    return tuple(_REGISTRY)


class SearchAgent(ABC):

    name = None
    model_id = None
    max_tokens = None
    services = ("AGENT",)
    credential = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _REGISTRY[cls.name] = cls

    def __init__(self, retriever, keys=None):
        self.retriever, self.keys = retriever, keys

    @property
    def key(self):
        return self.keys is not None

    def search_arguments(self):
        keys = self.keys.arguments() if self.keys else ()
        return ["--retriever", self.retriever, *keys]

    @property
    def prices(self):
        return PRICES[self.model_id]

    def check_service(self):
        from selfindex.runtime.platform.endpoints import endpoint

        for role in self.services:
            endpoint(role)
        if self.credential and not os.environ.get(self.credential):
            raise ValueError(f"{self.credential} is not set")

    def pre_agent(self, log_dir):  # noqa: B027
        ...

    def post_agent(self):  # noqa: B027
        ...

    @abstractmethod
    def agent_cmd(self, queries, out_dir, threads):
        raise NotImplementedError

    def client_arguments(self, queries, out_dir, threads):
        return [
            "--model",
            self.model_id,
            "--output-dir",
            out_dir,
            "--query",
            queries,
            "--num-threads",
            str(threads),
            "--max-tokens",
            str(self.max_tokens),
            *self.search_arguments(),
        ]

    def run_agent(self, queries, out_dir, threads, log_path):
        try:
            self.pre_agent(os.path.dirname(log_path) or ".")
            return self._run_process(queries, out_dir, threads, log_path)
        finally:
            self.post_agent()

    def _run_process(self, queries, out_dir, threads, log_path):
        os.makedirs(out_dir, exist_ok=True)
        cmd = self.agent_cmd(queries, out_dir, threads)
        print(
            f"[{self.name}] agent start · {' '.join(shlex.quote(c) for c in cmd[3:])}",
            flush=True,
        )
        with open(log_path, "a") as log:
            return subprocess.run(
                cmd, cwd=BROWSECOMP_DIR, env=base_env(), stdout=log, stderr=log
            ).returncode


class GptOss(SearchAgent):

    name = "gptoss"
    model_id = AGENT_MODEL
    max_tokens = 10000
    prices = (0.0, 0.0, 0.0)

    def check_service(self):
        from browsecomp.agent.ours_agent_common import agent_urls
        from selfindex.runtime.platform.endpoints import Endpoint

        super().check_service()
        for url in agent_urls():
            if (served := Endpoint(url).model) != self.model_id:
                raise ValueError(f"{url} serves {served}; the agent is {self.model_id}")

    def agent_cmd(self, queries, out_dir, threads):
        from browsecomp.agent.ours_agent_common import agent_urls

        return subprocess_argv(
            "bc.client.oss",
            [
                "--model-url",
                ",".join(agent_urls()),
                *self.client_arguments(queries, out_dir, threads),
            ],
        )


class Nano54(SearchAgent):

    name = "nano54"
    model_id = NANO_AGENT_MODEL
    max_tokens = 40000
    credential = API_KEY_ENV

    def agent_cmd(self, queries, out_dir, threads):
        return subprocess_argv(
            "bc.client.openai", self.client_arguments(queries, out_dir, threads)
        )


class Gem37(SearchAgent):

    name = "gem37"
    model_id = GEMINI_AGENT_MODEL
    max_tokens = 10000
    services = ()
    credential = "GEMINI_API_KEY"
    MCP_STARTUP_SECONDS = 1200
    MCP_POLL_SECONDS = 5

    def pre_agent(self, log_dir):
        self.port = _free_port()
        log_path = os.path.join(
            log_dir, f"gem37_mcp_{self.retriever}{'_key' if self.key else ''}.log"
        )
        with open(log_path, "w") as fh:
            self._mcp = subprocess.Popen(
                subprocess_argv(
                    "bc.mcp_server",
                    ["--port", str(self.port), *self.search_arguments()],
                ),
                cwd=BROWSECOMP_DIR,
                env=base_env(),
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        self._wait_for_mcp_server(log_path)

    def _wait_for_mcp_server(self, log_path):
        deadline = time.monotonic() + self.MCP_STARTUP_SECONDS
        while time.monotonic() < deadline:
            if self._mcp.poll() is not None:
                raise RuntimeError(f"MCP server died; see {log_path}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=2):
                    print(
                        f"[gem37] MCP server ready (:{self.port}/sse · "
                        f"{self.retriever} key={int(self.key)})",
                        flush=True,
                    )
                    return
            except OSError:
                time.sleep(self.MCP_POLL_SECONDS)
        raise RuntimeError(f"MCP server startup timed out; see {log_path}")

    def post_agent(self):
        if getattr(self, "_mcp", None) and self._mcp.poll() is None:
            self._mcp.kill()

    def agent_cmd(self, queries, out_dir, threads):
        return subprocess_argv(
            "bc.client.gemini",
            [
                "--model",
                self.model_id,
                "--mcp-url",
                f"http://127.0.0.1:{self.port}/sse",
                "--query",
                queries,
                "--output-dir",
                out_dir,
                "--num-threads",
                str(threads),
                "--max_tokens",
                str(self.max_tokens),
            ],
        )


class Kimi25(SearchAgent):

    name = "kimi25"
    model_id = KIMI_AGENT_MODEL
    max_tokens = 40000
    credential = or_client_ours.API_KEY_ENV

    def agent_cmd(self, queries, out_dir, threads):
        return subprocess_argv(
            "bc.client.or", self.client_arguments(queries, out_dir, threads)
        )


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get_backbone(name, **options):
    if name not in _REGISTRY:
        raise ValueError(
            f"unsupported BrowseComp agent: {name}; available: {', '.join(_REGISTRY)}"
        )
    return _REGISTRY[name](**options)


def format_metric(value, suffix=""):
    return "-" if value is None else f"{value:.2f}{suffix}"


def _run_judge(backbone, out_dir, log_dir):
    log_path = os.path.join(log_dir, f"judge_{os.path.basename(out_dir)}.log")
    with open(log_path, "w") as fh:
        returncode = subprocess.run(
            subprocess_argv("bc.judge_run", ["--input_dir", out_dir]),
            cwd=BROWSECOMP_DIR,
            env=base_env(),
            stdout=fh,
            stderr=subprocess.STDOUT,
        ).returncode
    if returncode != 0:
        raise RuntimeError(
            f"[{backbone.name}] judge failed (exit {returncode}); see {log_path}"
        )


def _summary_metrics(summary_path):
    summary = read_json(summary_path)
    tool_stats = summary.get("avg_tool_stats") or {}
    return {
        "n": len(summary.get("per_query_metrics") or []),
        "accuracy": summary.get("Accuracy (%)"),
        "recall": summary.get("Recall (%)"),
        "search_calls": (
            tool_stats.get("search") if isinstance(tool_stats, dict) else None
        ),
        "calibration_error": summary.get("Calibration Error (%)"),
    }


def _print_report(backbone, report):
    index = "evolved keys" if backbone.key else "base index"
    print(
        f"[BC+] {backbone.name}/{backbone.retriever} ({index})  "
        f"Accuracy {format_metric(report['accuracy'], '%')} · "
        f"Recall {format_metric(report['recall'], '%')} · "
        f"Search Calls {format_metric(report['search_calls'])} · "
        f"CalErr {format_metric(report['calibration_error'], '%')} · "
        f"n {report['n']} · ${report['cost_usd']:.2f}",
        flush=True,
    )


def judge_and_report(backbone, out_dir, log_dir, query_ids):
    from browsecomp.adapters.paths import evaluation_directory
    from browsecomp.pipeline.specification import verify_evaluation

    _run_judge(backbone, out_dir, log_dir)
    evaluation = evaluation_directory(out_dir)
    verify_evaluation(evaluation, query_ids)
    report = {
        "backbone": backbone.name,
        "model": backbone.model_id,
        "retriever": backbone.retriever,
        "key": int(backbone.key),
        "run_dir": out_dir,
        **_summary_metrics(evaluation / "evaluation_summary.json"),
    }
    usage = usage_summary(out_dir, backbone.prices)
    report["tokens"] = {
        k: usage.get(k) for k in ("input", "cached", "output", "reasoning")
    }
    report["cost_usd"] = round(usage.get("cost", 0.0), 4)
    write_json(evaluation / "final_report.json", report, indent=1, ensure_ascii=False)
    _print_report(backbone, report)

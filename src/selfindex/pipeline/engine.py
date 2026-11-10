from __future__ import annotations

import time
from collections import Counter

from selfindex.pipeline.contracts import Context, EditState, SignalSource
from selfindex.pipeline.optimizer import optimize
from selfindex.pipeline.settings import LoopSettings, parameter


def _timed(phases, name, action):
    started = time.time()
    result = action()
    phases[name] = round(time.time() - started, 1)
    return result


class OptimizerLoop:

    def __init__(
        self,
        ctx: Context,
        signal: SignalSource,
        settings: LoopSettings,
    ):
        self.ctx = ctx
        self.signal = signal
        self.eligible = set(ctx.eligible)
        self.state = EditState()
        self.settings = settings
        self.validation = Counter()
        self.round_log = []

    def run(self):
        rounds = self.settings.rounds
        for round_index in range(rounds):
            before = len(self.state.edited)
            before_revised = int(self.state.revised)
            before_keys = int(self.state.keys)
            print(
                f"\n===== ROUND {round_index + 1}/{rounds} (documents with keys {before}) =====",
                flush=True,
            )
            started = time.time()
            self.ctx.round_state.round = round_index
            report = self._run_round()
            self._finish_round(
                round_index, before, started, before_revised, before_keys
            )
        report["rounds"] = rounds
        return report

    def _run_round(self):
        state, ctx = self.state, self.ctx
        if not state.t0:
            state.t0 = time.time()
        phases = self._phases = {}
        sig = _timed(phases, "signal", lambda: self.signal.build(ctx))
        queries = list((sig.meta or {}).get("queries") or [])
        replacements, report = _timed(
            phases, "optimize", lambda: self._optimize(queries)
        )
        _timed(phases, "commit", lambda: self._commit(replacements))
        state.processed += report.diagnosed
        self.validation.update(report.counters)
        print(
            f"[optimizer] {len(queries):,} queries | feedback on {report.feedback_documents:,} documents | "
            f"diagnosed {report.diagnosed:,}, shortfall {report.shortfalls:,} | first keys {report.first_keys:,} | "
            f"proposals {report.proposals:,} | replaced {len(replacements):,} | "
            + ", ".join(f"{k} {v:,}" for k, v in sorted(report.counters.items())),
            flush=True,
        )
        return self._round_report()

    def _optimize(self, queries):
        ctx = self.ctx
        per_document = parameter("MAX_KEYS") - 1
        with ctx.retriever.snapshot(ctx.texts) as view:
            return optimize(
                ctx.llm,
                view,
                ctx.texts,
                queries,
                depth=parameter("RETRIEVAL_DEPTH"),
                eligible=self.eligible,
                per_document=per_document,
            )

    def _commit(self, replacements):
        index, state = self.ctx.index, self.state
        held = {
            doc: len((index.generated_keys or {}).get(doc) or [])
            for doc in replacements
        }
        index.replace_validated_many(replacements)
        for doc, keys in replacements.items():
            if doc in state.edited:
                state.revised += 1
            state.edited.add(doc)
            index.accepted_edits[doc] = max(1, index.accepted_edits[doc])
            state.committed += 1
            state.keys += len(keys) - held[doc]

    def _finish_round(self, r, before, started, before_revised, before_keys):
        seconds = round(time.time() - started, 1)
        phases = dict(self._phases)
        phases.update(
            {
                k: round(v, 1)
                for k, v in (self.ctx.round_state.phase_seconds or {}).items()
            }
        )
        self.ctx.round_state.phase_seconds = {}
        gained = len(self.state.edited) - before
        coverage = 100.0 * len(self.state.edited) / max(len(self.ctx.texts), 1)
        record = {
            "round": r + 1,
            "gained": gained,
            "revised": int(self.state.revised) - int(before_revised),
            "keys_added": int(self.state.keys) - int(before_keys),
            "seconds": seconds,
            "edited": len(self.state.edited),
            "coverage_pct": round(coverage, 2),
            "keys": sum(len(keys) for keys in self.ctx.index.generated_keys.values()),
        }
        record.update(self._usage_record())
        record["phases"] = phases
        record["round_end_wall"] = round(time.time(), 1)
        self.round_log.append(record)
        print(
            "[round:phases] " + ", ".join(f"{k} {v}s" for k, v in phases.items()),
            flush=True,
        )
        print(
            f"[round {r + 1}] new {gained}, revised {record['revised']}, documents with keys "
            f"{len(self.state.edited)} · coverage {coverage:.1f}% · {seconds:.0f}s",
            flush=True,
        )

    def _usage_record(self):
        usage = getattr(getattr(self.ctx, "llm", None), "tok", None) or {}
        return {
            "llm_calls": int(usage.get("calls", 0)),
            "llm_prompt_tokens": int(usage.get("prompt", 0)),
            "llm_completion_tokens": int(usage.get("completion", 0)),
            "llm_cached_tokens": int(usage.get("cached", 0)),
            "wall_sec": round(time.time() - (self.state.t0 or time.time()), 1),
        }

    def _round_report(self):
        state = self.state
        return {
            "processed": state.processed,
            "committed": state.committed,
            "revised": state.revised,
            "keys": state.keys,
            "edited_docs": len(state.edited),
            "seconds": round(time.time() - state.t0, 1),
            "slots": {"signal": self.signal.name},
        }

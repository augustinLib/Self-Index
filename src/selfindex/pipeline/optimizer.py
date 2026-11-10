from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from selfindex.common.judge_prompts import FAITHFULNESS
from selfindex.common.optimizer_prompts import DIAGNOSIS, REVISION
from selfindex.common.parsing import extract_obj
from selfindex.pipeline.generation.faithfulness import (
    FAITHFULNESS_TOKEN_OVERHEAD,
    FAITHFULNESS_TOKENS_PER_KEY,
    parse_faithfulness,
)
from selfindex.revision.base import ProposedKey
from selfindex.validation.base import RevisionDecision
from selfindex.validation.validator import validate_scores

DOCUMENT_CHARS = 2400
KEY_CHARS = 200
FEEDBACK_QUERIES = 10
KEYS_ABOVE = 3
PROFILE_KEYS = 3
COMPETING_KEYS = 20
KEYS_PER_REQUEST = 9
DIAGNOSIS_TEMPERATURE = 0.5
REVISION_TEMPERATURE = 0.5
FAITHFULNESS_TEMPERATURE = 0.3
DIAGNOSIS_TOKENS = 1200
REVISION_TOKENS = 500
KEY_TOKENS = 200


@dataclass
class RetrievalFeedback:

    queries: list[str]
    key_ids: object
    scores: object
    profiles: dict[int, Counter] = field(default_factory=dict)
    records: dict[int, list[int]] = field(default_factory=dict)


def collect_feedback(view, queries, depth) -> RetrievalFeedback:
    result = view.retrieve_keys(list(queries), depth)
    feedback = RetrievalFeedback(list(queries), result.key_ids, result.scores)
    for query, row in enumerate(result.key_ids):
        kids = [int(kid) for kid in row]
        owners = [view.key_owner(kid) for kid in kids]
        for kid, owner in zip(kids, owners):
            profile = feedback.profiles.setdefault(kid, Counter())
            profile.update(
                other
                for other, other_owner in zip(kids, owners)
                if other_owner != owner
            )
            records = feedback.records.setdefault(owner, [])
            if not records or records[-1] != query:
                records.append(query)
    return feedback


def _clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _key_label(kid, current):
    return f"key {current.index(kid) + 1}"


def format_feedback(view, feedback, doc, current):
    own = set(current)
    shown = []
    for query in feedback.records.get(doc, []):
        ranks = [
            rank for rank, kid in enumerate(feedback.key_ids[query]) if int(kid) in own
        ]
        shown.append((ranks[0], query, ranks))
    shown.sort()
    lines = []
    for number, (best, query, ranks) in enumerate(shown[:FEEDBACK_QUERIES], 1):
        lines.append(f"Query {number}: {_clip(feedback.queries[query], 400)}")
        for rank in range(min(best, KEYS_ABOVE)):
            kid = int(feedback.key_ids[query][rank])
            lines.append(
                f"  rank {rank + 1}, score {feedback.scores[query][rank]:.3f}, "
                f"another document's key: {_clip(view.key_text(kid), KEY_CHARS)}"
            )
        for rank in ranks:
            kid = int(feedback.key_ids[query][rank])
            lines.append(
                f"  rank {rank + 1}, score {feedback.scores[query][rank]:.3f}, "
                f"this document's {_key_label(kid, current)}"
            )
    lines.append(
        "Co-retrieval (keys of other documents retrieved with this document's keys):"
    )
    for kid in current:
        profile = feedback.profiles.get(kid)
        if not profile:
            continue
        lines.append(f"  {_key_label(kid, current)}:")
        for other, count in profile.most_common(PROFILE_KEYS):
            lines.append(f"    {count}x {_clip(view.key_text(other), KEY_CHARS)}")
    return "\n".join(lines)


def _numbered(texts):
    return "\n".join(f"{i}. {text}" for i, text in enumerate(texts, 1)) or "(none)"


def diagnosis_message(view, texts, feedback, doc):
    current = [-(doc + 1), *view.keys_of(doc)]
    keys = ["the original text of the document, shown above"] + [
        view.key_text(kid) for kid in current[1:]
    ]
    content = DIAGNOSIS.format(
        doc=texts[doc][:DOCUMENT_CHARS],
        keys=_numbered(keys),
        feedback=format_feedback(view, feedback, doc, current),
    )
    return [{"role": "user", "content": content}]


@dataclass(frozen=True)
class Diagnosis:

    causes: dict[int, str]
    key_set: tuple[str, ...]
    revision: str

    @property
    def shortfall(self):
        return bool(self.causes or self.key_set)


def parse_diagnosis(raw, current) -> Diagnosis:
    reply = extract_obj(raw)
    causes, key_set = {}, []
    for entry in reply.get("diagnoses") or []:
        if not isinstance(entry, dict):
            continue
        cause = " ".join(str(entry.get("cause") or "").split())
        key = entry.get("key")
        if isinstance(key, str) and key.strip().lower().replace(" ", "_") == "key_set":
            key_set.append(cause)
            continue
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= len(current):
            kid = current[number - 1]
            causes[kid] = "; ".join(filter(None, (causes.get(kid), cause)))
    revision = " ".join(str(reply.get("revision") or "").split())
    return Diagnosis(causes, tuple(filter(None, key_set)), revision)


def competing_keys(feedback, diagnosis, current):
    implicated = current if diagnosis.key_set else list(diagnosis.causes)
    union = Counter()
    for kid in implicated:
        union.update(feedback.profiles.get(kid) or {})
    return [kid for kid, _ in union.most_common()]


def revision_message(view, texts, doc, diagnosis, competitors, room):
    generated = list(view.keys_of(doc))
    keys = ["the original text of the document, shown above (always kept)"]
    for kid in generated:
        text = view.key_text(kid)
        if cause := (diagnosis.causes.get(kid) if diagnosis else None):
            text += f" [falls short: {cause}]"
        keys.append(text)
    guidance = []
    if diagnosis is not None:
        if diagnosis.revision:
            guidance.append(diagnosis.revision)
        if original := diagnosis.causes.get(-(doc + 1)):
            guidance.append(f"The original text as a key: {original}")
        guidance.extend(
            f"The key set as a whole: {cause}" for cause in diagnosis.key_set
        )
    neighbors = [_clip(view.key_text(kid), 300) for kid in competitors[:COMPETING_KEYS]]
    content = REVISION.format(
        doc=texts[doc][:DOCUMENT_CHARS],
        keys=_numbered(keys),
        diagnosis="\n".join(guidance) or "(none)",
        neighbors="\n".join(f"- {text}" for text in neighbors) or "(none)",
        k=room,
    )
    return [{"role": "user", "content": content}], generated


def _key_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def parse_revision(raw, generated, view, room, original):
    reply = extract_obj(raw)
    if not reply:
        return None
    existing = {_key_text(view.key_text(kid)): kid for kid in generated}
    keep, numbers, revised = set(), reply.get("keep"), reply.get("revised")
    for number in numbers if isinstance(numbers, list) else []:
        try:
            index = int(number) - 2
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(generated):
            keep.add(generated[index])
    new, seen = [], {_key_text(original)}
    for text in revised if isinstance(revised, list) else []:
        text = _key_text(text)
        if text in existing:
            keep.add(existing[text])
        elif text and text not in seen:
            seen.add(text)
            new.append(text)
    kept = [kid for kid in generated if kid in keep]
    keys = [ProposedKey(view.key_text(kid), kid) for kid in kept]
    keys += [ProposedKey(text) for text in new[:room]]
    return keys


def faithfulness_message(texts, doc, keys):
    content = FAITHFULNESS.format(
        doc=texts[doc][:DOCUMENT_CHARS], views=_numbered(key.text for key in keys)
    )
    return [{"role": "user", "content": content}]


@dataclass
class IterationReport:
    feedback_documents: int = 0
    diagnosed: int = 0
    shortfalls: int = 0
    first_keys: int = 0
    proposals: int = 0
    counters: Counter = field(default_factory=Counter)


def optimize(llm, view, texts, queries, *, depth, eligible, per_document):
    report = IterationReport()
    feedback = collect_feedback(view, queries, depth)
    targets = sorted(
        doc for doc in feedback.records if view.keys_of(doc) or doc in eligible
    )
    report.feedback_documents = len(feedback.records)
    report.diagnosed = len(targets)
    raws = llm.batch(
        [diagnosis_message(view, texts, feedback, doc) for doc in targets],
        temperature=DIAGNOSIS_TEMPERATURE,
        max_tokens=DIAGNOSIS_TOKENS,
    )
    diagnoses, competitors = {}, {}
    for doc, raw in zip(targets, raws):
        current = [-(doc + 1), *view.keys_of(doc)]
        diagnosis = parse_diagnosis(raw, current)
        if diagnosis.shortfall:
            diagnoses[doc] = diagnosis
            competitors[doc] = competing_keys(feedback, diagnosis, current)
    report.shortfalls = len(diagnoses)
    report.first_keys = sum(1 for doc in diagnoses if not view.keys_of(doc))
    room = min(KEYS_PER_REQUEST, per_document)
    revise = sorted(diagnoses)
    messages, generated = [], {}
    for doc in revise:
        message, generated[doc] = revision_message(
            view, texts, doc, diagnoses.get(doc), competitors.get(doc, []), room
        )
        messages.append(message)
    raws = llm.batch(
        messages,
        temperature=REVISION_TEMPERATURE,
        per_job=[{"max_tokens": REVISION_TOKENS + KEY_TOKENS * room} for _ in messages],
    )
    proposals = {}
    for doc, raw in zip(revise, raws):
        keys = parse_revision(raw, generated[doc], view, room, texts[doc]) or []
        if len(keys) > per_document:
            report.counters["over_limit_replies"] += 1
            new = [key for key in keys if key.is_new][:per_document]
            kept = [key for key in keys if not key.is_new]
            keys = kept[: per_document - len(new)] + new
        if any(key.is_new for key in keys):
            proposals[doc] = keys
    report.proposals = len(proposals)
    return _validate(llm, view, texts, proposals, competitors, depth, report)


def _validate(llm, view, texts, proposals, competitors, depth, report):
    docs = sorted(proposals)
    raws = llm.batch(
        [faithfulness_message(texts, doc, proposals[doc]) for doc in docs],
        temperature=FAITHFULNESS_TEMPERATURE,
        per_job=[
            {
                "max_tokens": FAITHFULNESS_TOKEN_OVERHEAD
                + FAITHFULNESS_TOKENS_PER_KEY * len(proposals[doc])
            }
            for doc in docs
        ],
    )
    faithful = {}
    for doc, raw in zip(docs, raws):
        passed = parse_faithfulness(raw, len(proposals[doc]))
        faithful[doc] = [key for i, key in enumerate(proposals[doc]) if i in passed]
        report.counters["faithfulness_dropped"] += len(proposals[doc]) - len(
            faithful[doc]
        )
    kept, counters = validate_scores(
        view,
        {doc: [key.text for key in keys] for doc, keys in faithful.items() if keys},
        {doc: tuple(view.keys_of(doc)) for doc in docs},
        {doc: tuple(competitors.get(doc, ())) for doc in docs},
        depth=depth,
    )
    report.counters.update(counters)
    replacements = {}
    for doc in docs:
        accepted = set(kept.get(doc, []))
        decision = RevisionDecision(
            doc, tuple(key for key in faithful[doc] if key.text in accepted)
        )
        if decision.replacement is not None:
            replacements[doc] = list(decision.replacement)
    report.counters["replaced"] = len(replacements)
    return replacements, report

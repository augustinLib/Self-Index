from __future__ import annotations

from selfindex.common.parsing import extract_list, extract_obj
from selfindex.settings import DOCUMENT_CLIP_CHARS

_ABSTRACTION_PROMPT = """Read the document and state, in plain language, the underlying problem, phenomenon or question that
this document resolves for a reader. Describe it the way a person who has NOT read the document would
experience or wonder about it: what they observe, want, or are confused by. Do not name the document’s
specialized terms, entities or titles.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Answer JSON only:
{{"problem": "..."}}"""

_QUERY_PROMPT = """You are a person in the situation below. You have NOT read any document about it and you do not know
its technical vocabulary. Write {m} different questions you would post on a Q&A forum: your concrete
observation or goal, what puzzles you, and what you have tried or assumed. Write in everyday language,
and make each question a different need or a different way of asking.
Situation:
{problem}
Answer JSON only:
{{"queries": ["...", "..."]}}"""


QUERY_TEMPERATURE = 0.8
QUERIES_PER_DOCUMENT = 1


class QueryGenerator:

    def __init__(self, llm, doc_texts):
        self.llm = llm
        self.texts = doc_texts

    def _batch(self, contents):
        return self.llm.batch(
            [[{"role": "user", "content": content}] for content in contents],
            temperature=QUERY_TEMPERATURE,
            max_tokens=300,
        )

    def generate_batch(self, docs) -> dict[int, list[str]]:
        docs = list(docs)
        raws = self._batch(
            _ABSTRACTION_PROMPT.format(doc=self.texts[d][:DOCUMENT_CLIP_CHARS])
            for d in docs
        )
        problems = {}
        for d, raw in zip(docs, raws):
            problem = extract_obj(raw).get("problem")
            if isinstance(problem, str) and problem.strip():
                problems[d] = problem.strip()
        raws = self._batch(
            _QUERY_PROMPT.format(m=QUERIES_PER_DOCUMENT, problem=problem)
            for problem in problems.values()
        )
        out = {}
        for d, raw in zip(problems, raws):
            queries = extract_list(raw, "queries")
            out[d] = [q.strip() for q in queries if isinstance(q, str) and q.strip()][
                :QUERIES_PER_DOCUMENT
            ]
        return out

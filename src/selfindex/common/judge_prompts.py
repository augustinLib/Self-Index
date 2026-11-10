SUPPORT = """• First identify the information need behind it. A search may be a question, a how-to, a bare keyword
phrase, a claim to verify, or a scenario with a persona (“I’m a nurse who . . . ”). Judge the need, not the
wording or the framing. Searchers use everyday words where the document uses technical ones, and
that is fine.
• The need is RESOLVED (YES) when the document itself contains what the searcher needs: the answer,
the explanation or mechanism, the procedure or technique, or the definition. It is also resolved when the
document states a fact or principle from which the answer follows with ordinary reasoning. A claim is
resolved when the document states, supports, or refutes it. A keyword or topic search is resolved when
the document substantively covers that topic. A search for a source, study, or overview is resolved when
this document is that source.
• The need is NOT resolved (NO) when the document only shares the topic or mentions it in passing,
when the need turns on a specific detail the document lacks (a reason why, a comparison, a number, a
step-by-step procedure, a result or outcome), or when the document is only a title, heading, citation
entry, reference list, table fragment, or figure caption that names the subject without containing the
content."""


SCALE = """Scale:
0 the document has nothing to do with the need
1 related topic but the need is not met
2 the document meets the need, even if the answer is partial, implicit, or amid other material
3 the document is dedicated to the need and contains the exact answer"""


FAITHFULNESS = (
    """You are auditing a search index. Below is one document and a numbered list of SEARCHES written so that
this document can be found. For each search rate how well THIS document would resolve it for the user
who typed it.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Searches:
{views}
How to judge each search:
"""
    + SUPPORT
    + """
• Judge each search on its own. The verdict on one search must not influence another.
"""
    + SCALE
    + """
Reply format: one line per search, in order: the number, a colon, at most twelve words naming the need
and what the document offers for it, then => and the score. Nothing else.
Example: 3: asks why, document lists causes only => 1."""
)


ANSWERABILITY = (
    """You are auditing a search index. Below is one document and one SEARCH written so that this document
can be found. Rate how well THIS document would resolve it for the user who typed it.
Document:
<<<DOC START>>>
{doc}
<<<DOC END>>>
Search:
{query}
How to judge the search:
"""
    + SUPPORT
    + "\n"
    + SCALE
    + """
Reply format: one line: the score, then a dash and at most ten words of reason. Nothing else."""
)

PASSING_SCORE = 2

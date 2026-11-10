DIAGNOSIS = """You maintain the search index entry of the document below. The entry is a set of KEYS: texts the search
engine matches against user queries. A query retrieves the document through whichever of its keys matches
best. The retrieval feedback below shows how the current keys behave for observed queries.
DOCUMENT:
<<<DOC START>>>
{doc}
<<<DOC END>>>
CURRENT KEYS (numbered):
{keys}
RETRIEVAL FEEDBACK (query texts, retrieved keys and their scores, and cross-document co-retrieval
frequencies):
{feedback}
Read the document and examine its current keys together with the retrieval feedback. Determine whether
the current representations have any shortfalls that warrant revision. Multiple documents may satisfy the
same query, so co-retrieval alone does not establish a shortfall.
Identify the current key responsible for each shortfall. If the shortfall concerns insufficient coverage by the
key set as a whole rather than a particular key, use “key set”. Explain what the current representations fail
to express or distinguish. Then state concretely what a revision of the key set should change. Do not write
the revised keys.
If no shortfall is supported, return an empty diagnoses list and an empty revision string.
Answer JSON only:
{{"diagnoses": [
   {{"key": <number or "key_set">,
    "cause": "<why it falls short>"}}
],
 "revision": "<what to change>"}}"""


REVISION = """You maintain the search index entry of the document below: a set of KEYS, short texts the search engine
matches against user queries. A query retrieves the document through whichever key matches best.
DOCUMENT:
<<<DOC START>>>
{doc}
<<<DOC END>>>
CURRENT KEYS (numbered; a diagnosed key is marked with what falls short):
{keys}
SUGGESTED CHANGES FROM SELF-DIAGNOSIS:
{diagnosis}
COMPETING KEYS (index entries of other documents retrieved for the same queries; use these to
identify what the current document provides that its keys should distinguish more clearly):
{neighbors}
Revise the key set as a whole. Keep the keys that still serve the document, drop or rewrite the diagnosed
ones, and write at most {k} new or rewritten keys so that anyone who needs what this document provides
can find it, whatever they already know and however they would ask. The kept and new keys together number
at most {k}. Write each key in the words such a searcher would use, not in the document’s words. Decide the
form and length of each key yourself, from the document and from how people would look for it.
When the document has no keys yet, start from the common ways people look for documents like this one:
describing their situation or problem in their own words, asking why or how something happens, checking a
claim, typing a few keywords, asking how to do a task, comparing alternatives, or bringing a related problem
that this document helps solve.
Each key must serve a need, a situation, or a way of asking that the other keys do not. Do not restate or
copy the document; do not repeat a kept key.
Every key you write must satisfy three requirements:
• FAITHFUL: it describes only knowledge this document actually contains or resolves; nothing invented,
  nothing the document does not support.
• SPECIFIC: it captures what is specific to THIS document, not content broadly shared with other
  documents in the corpus.
• SEPARATED: it reduces ambiguity with the competing keys by emphasizing distinctions supported by
  this document. A shared information need is acceptable when this document also addresses it.
Answer JSON only:
{{"keep": [<numbers of keys to keep>],
 "revised": ["<key>", ...]}}"""

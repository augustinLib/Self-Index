from selfindex.validation.base import ScoreCriterion, _finite_scores


class SeparationCriterion(ScoreCriterion):
    def __init__(self, live, previous_texts, competitors, encoded):
        self.live = live
        self.previous_texts = previous_texts
        self.competitors = competitors
        self.encoded = encoded

    def _columns(self, doc, column_map, device):
        import torch

        return torch.as_tensor(
            [column_map[int(key)] for key in self.competitors[doc]], device=device
        )

    def apply(self, chunk, keep, stats):
        import torch

        owners = chunk.owners
        docs = sorted({doc for doc in owners if self.competitors.get(doc)})
        stats["separation_vacuous"] += sum(
            not self.competitors.get(doc) for doc in set(owners)
        )
        if not docs:
            return
        union = sorted({int(key) for doc in docs for key in self.competitors[doc]})
        column_map = {key: column for column, key in enumerate(union)}
        checked = set(docs)
        texts = [text for text, owner in zip(chunk.texts, owners) if owner in checked]
        texts += [text for doc in docs for text in self.previous_texts.get(doc, ())]
        row_of = {text: row for row, text in enumerate(dict.fromkeys(texts))}
        scores = _finite_scores(self.live.key_scores(self.encoded(list(row_of)), union))
        originals = _finite_scores(
            self.live.key_scores(self.live.doc_query_repr(docs), union)
        )
        kept = keep.tolist()
        for position, doc in enumerate(docs):
            cols = self._columns(doc, column_map, scores.device)
            baseline = float(originals[position, cols].max().item())
            if previous := [row_of[text] for text in self.previous_texts.get(doc, ())]:
                baseline = max(baseline, float(scores[previous][:, cols].max().item()))
            rows = [i for i, owner in enumerate(owners) if owner == doc and kept[i]]
            if not rows:
                continue
            proposed = [row_of[chunk.texts[i]] for i in rows]
            rejected = scores[proposed][:, cols].max(1).values >= baseline
            stats["separation_checked"] += len(rows)
            stats["separation_dropped"] += int(rejected.sum().item())
            keep[torch.as_tensor(rows, device=scores.device)[rejected]] = False

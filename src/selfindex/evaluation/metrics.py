import numpy as np


class NDCGMetric:
    name = "ndcg"
    scale = 100.0
    k = 10

    @property
    def label(self):
        return f"nDCG@{self.k}"

    def values(self, scores, golds, exclusions):
        return ndcg_values(scores, golds, exclusions, self.k)


def ndcg_values(scores_rows, golds, excl, k):
    values = []
    for i, gold in enumerate(golds):
        if not gold:
            continue
        ranked = _ranked(scores_rows[i], excl[i], k)
        dcg = sum(1.0 / np.log2(r + 2) for r, d in enumerate(ranked) if d in gold)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(gold))))
        values.append(dcg / idcg if idcg else 0.0)
    return values


def _ranked(row, excluded, k):
    depth = min(len(row) - 1, k + len(excluded))
    part = np.argpartition(-row, depth)[: depth + 1]
    selected = row[part]
    if (
        len(np.unique(selected)) == len(selected)
        and np.count_nonzero(row == selected.min()) == 1
    ):
        part = part[np.argsort(-row[part])]
        return [int(d) for d in part if int(d) not in excluded][:k]
    return [int(d) for d in np.argsort(-row) if int(d) not in excluded][:k]

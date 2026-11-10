from selfindex.validation.base import ScoreCriterion, _finite_scores


class SpecificityCriterion(ScoreCriterion):
    def __init__(self, live, depth):
        self.live = live
        self.depth = depth

    def apply(self, chunk, keep, stats):
        import torch

        scores = _finite_scores(self.live.original_scores(chunk.queries))
        docs = torch.as_tensor(chunk.owners, device=scores.device)
        own_scores = scores[torch.arange(len(docs), device=scores.device), docs]
        rank = (scores >= own_scores.unsqueeze(1)).sum(1) - 1
        accepted = rank < self.depth
        stats["specificity_dropped"] += int((~accepted).sum().item())
        keep &= accepted

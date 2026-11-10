import numpy as np

from selfindex.runtime.platform.devices import compute_device


def fold_max_np(S0, key_scores, owner):
    folded = owner_amax(key_scores, owner, S0.shape[1])
    return np.maximum(S0, folded)


def owner_amax(sv, owner, n_docs):
    import torch

    dev = compute_device()
    with torch.no_grad():
        scores = torch.as_tensor(
            np.ascontiguousarray(sv), dtype=torch.float32, device=dev
        )
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        owners = torch.as_tensor(
            np.ascontiguousarray(owner), dtype=torch.int64, device=dev
        )
        folded = torch.full(
            (scores.shape[0], n_docs), float("-inf"), dtype=torch.float32, device=dev
        )
        if owners.numel():
            folded.scatter_reduce_(
                1,
                owners.unsqueeze(0).expand(scores.shape[0], -1),
                scores,
                reduce="amax",
            )
        return folded.cpu().numpy()

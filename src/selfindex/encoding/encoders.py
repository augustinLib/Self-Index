from __future__ import annotations

import numpy as np

from selfindex.common.models.encoders import END_OF_TEXT_TOKEN
from selfindex.runtime.platform.devices import compute_device
from selfindex.runtime.platform.settings import ENCODE_BATCH, encoder_settings

_MODELS = {}


def _load(dev):
    settings = encoder_settings()
    if settings.model not in _MODELS:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(settings.model)
        model = (
            AutoModel.from_pretrained(
                settings.model,
                torch_dtype=(
                    torch.float32 if dev == "cpu" else getattr(torch, settings.dtype)
                ),
            )
            .to(dev)
            .eval()
        )
        _MODELS[settings.model] = (tok, model)
    return _MODELS[settings.model]


def _tok_chunk(tok, chunk, maxlen, dev):
    if encoder_settings().append_eod:
        encoded = tok(chunk, padding=False, truncation=True, max_length=maxlen - 2)
        eod = tok.convert_tokens_to_ids(END_OF_TEXT_TOKEN)
        for ids, mask in zip(
            encoded["input_ids"], encoded["attention_mask"], strict=False
        ):
            ids.append(eod)
            mask.append(1)
        return tok.pad(encoded, padding=True, return_tensors="pt").to(dev)
    return tok(
        chunk, padding=True, truncation=True, max_length=maxlen, return_tensors="pt"
    ).to(dev)


def _tokenize(tok, chunk, maxlen, dev):
    try:
        return chunk, _tok_chunk(tok, chunk, maxlen, dev)
    except Exception:
        chunk = [
            str(text).encode("utf-8", "ignore").decode("utf-8", "ignore") or " "
            for text in chunk
        ]
        return chunk, _tok_chunk(tok, chunk, maxlen, dev)


def _pooled(hidden, encoded):
    import torch

    if encoder_settings().pooling == "cls":
        vectors = hidden[:, 0]
    else:
        last = encoded["attention_mask"].sum(dim=1) - 1
        vectors = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
    return torch.nn.functional.normalize(vectors.float(), p=2, dim=1)


def _encode_on(texts, dev, is_query, batch, maxlen):
    import torch

    tok, model = _load(dev)
    if is_query:
        texts = [encoder_settings().query_prefix + t for t in texts]
    out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        _encode_chunks(tok, model, texts, dev, batch, maxlen, out)
    return out


def _encode_chunks(tok, model, texts, dev, batch, maxlen, out):
    start = 0
    while start < len(texts):
        chunk, encoded = _tokenize(tok, texts[start : start + batch], maxlen, dev)
        hidden = model(**encoded)[0]
        out[start : start + len(chunk)] = _pooled(hidden, encoded).cpu().numpy()
        start += len(chunk)


def _as_strings(texts):
    if any(type(t) is not str for t in texts):
        return [t if type(t) is str else ("" if t is None else str(t)) for t in texts]
    return texts


def _length_order(texts, batch):
    n = len(texts)
    if n > batch:
        return np.argsort(
            np.fromiter((len(t) for t in texts), np.int64, n), kind="stable"
        )
    return np.arange(n)


def encode(texts: list[str], is_query: bool = False) -> np.ndarray:
    settings = encoder_settings()
    texts = _as_strings(texts)
    if len(texts) == 0:
        return np.zeros((0, settings.dimension), dtype=np.float32)
    order = _length_order(texts, ENCODE_BATCH)
    embeddings = _encode_on(
        [texts[i] for i in order],
        compute_device(),
        is_query,
        ENCODE_BATCH,
        settings.max_length,
    )
    out = np.empty_like(embeddings)
    out[order] = embeddings
    return out

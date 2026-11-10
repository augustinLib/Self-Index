import argparse
import json
import os
import time

import numpy as np
import torch

from browsecomp.adapters.embedding import QwenEmbBackend, dense_encoder_options
from browsecomp.agent._paths import BROWSECOMP_DIR
from selfindex.common.io import read_lines
from selfindex.common.models.roles import encoder_model
from selfindex.runtime.platform.devices import compute_device
from selfindex.runtime.platform.environment import ensure_env

BATCH_SIZE = 256
TOKEN_BUDGET_PER_ITEM = 512
TOKEN_BUDGET = BATCH_SIZE * TOKEN_BUDGET_PER_ITEM
MAX_SEQUENCE_LENGTH = 8192
CHARACTERS_PER_TOKEN = 3
TOKEN_ESTIMATE_OVERHEAD = 8
PROGRESS_INTERVAL = 20


def encoding_spans(keys):
    spans, start = [], 0
    while start < len(keys):
        longest, end = 0, start
        while end < len(keys):
            length = min(
                len(keys[end]) // CHARACTERS_PER_TOKEN + TOKEN_ESTIMATE_OVERHEAD,
                MAX_SEQUENCE_LENGTH,
            )
            padded_length = max(longest, length)
            if end > start and padded_length * (end - start + 1) > TOKEN_BUDGET:
                break
            longest, end = padded_length, end + 1
        spans.append((start, end))
        start = end
    return spans


def _load_key_pairs(directory):
    return [
        (record["k"], str(record["docid"]))
        for record in (
            json.loads(line)
            for line in read_lines(os.path.join(directory, "keys.jsonl"))
        )
    ]


def _encode_spans(model, tokenizer, keys, spans, device, started):
    batches = []
    with torch.no_grad():
        for index, (start, end) in enumerate(spans):
            inputs = tokenizer(
                keys[start:end],
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.amp.autocast("cuda", enabled=device != "cpu"):
                encoded = model.encode_passage(inputs)
            batches.append(encoded.cpu().float().numpy().astype(np.float16))
            if index % PROGRESS_INTERVAL == 0:
                print(
                    f"[enc] {end:,}/{len(keys):,} ({end / max(1, time.time() - started):.0f}/s)",
                    flush=True,
                )
    return batches


def _encode_by_length(keys, device):
    from tevatron.retriever.driver.encode import DenseModel
    from transformers import AutoTokenizer

    model = (
        DenseModel.load(
            encoder_model(), **dense_encoder_options(device, ensure_env().hf_home)
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_model(), cache_dir=ensure_env().hf_home, padding_side="left"
    )
    order = np.argsort(
        np.fromiter((len(k) for k in keys), np.int64, len(keys)), kind="stable"
    )
    inverse = np.empty(len(order), np.int64)
    inverse[order] = np.arange(len(order))
    keys = [keys[i] for i in order]
    spans = encoding_spans(keys)
    print(
        f"[enc] token budget {TOKEN_BUDGET:,} -> {len(spans):,} batches "
        f"(mean {len(keys) / max(1, len(spans)):.0f} keys)",
        flush=True,
    )
    started = time.time()
    batches = _encode_spans(model, tokenizer, keys, spans, device, started)
    return np.concatenate(batches)[inverse], started


def _zero_nan_rows(vectors):
    bad = np.where(np.isnan(vectors).any(axis=1))[0]
    if len(bad):
        vectors[bad] = 0
        print(f"[enc] NaN {len(bad)} rows zeroed", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keys",
        required=True,
        help="key-set directory (relative: under adapters/browsecomp/)",
    )
    directory = os.path.join(BROWSECOMP_DIR, parser.parse_args(argv).keys)
    torch.backends.cuda.enable_cudnn_sdp(False)
    device = compute_device()
    pairs = _load_key_pairs(directory)
    keys = [k for k, _ in pairs]
    if not keys:
        raise ValueError("No keys to encode; check the key export")
    print(f"[enc] keys {len(keys):,}", flush=True)

    row_of = {d: i for i, d in enumerate(QwenEmbBackend().lookup)}
    owner = np.asarray([row_of[d] for _, d in pairs], np.int64)
    vectors, started = _encode_by_length(keys, device)
    _zero_nan_rows(vectors)
    np.savez(os.path.join(directory, "key_emb.npz"), v=vectors, owner=owner)
    print(f"[enc] done {vectors.shape} ({time.time() - started:.0f}s)", flush=True)

from __future__ import annotations

from dataclasses import dataclass, replace

from selfindex.common.models.encoders import ENCODERS

BM25_K1 = 0.9
BM25_B = 0.4
ENCODE_BATCH = 512
CORPUS_SETTINGS = ("max_length", "query_prefix")


@dataclass(frozen=True)
class EncoderSettings:
    name: str
    model: str
    dimension: int
    pooling: str
    dtype: str
    max_length: int
    append_eod: bool
    query_prefix: str

    @classmethod
    def of(cls, name, **corpus_settings):
        if name not in ENCODERS:
            raise ValueError(f"unknown encoder {name} (choose from {list(ENCODERS)})")
        if unknown := sorted(set(corpus_settings) - set(CORPUS_SETTINGS)):
            raise ValueError(f"a corpus cannot set the encoder's {unknown}")
        values = ENCODERS[name]
        table = cls(
            name,
            values["model"],
            values["dim"],
            values["pooling"],
            values["dtype"],
            values["maxlen"],
            values["append_eod"],
            values["query_prefix"],
        )
        return replace(table, **corpus_settings)


_SELECTED: EncoderSettings | None = None


def use_encoder(settings: EncoderSettings) -> EncoderSettings:
    global _SELECTED
    _SELECTED = settings
    return settings


def encoder_settings() -> EncoderSettings:
    if _SELECTED is None:
        raise RuntimeError("no encoder selected: the pipeline selects it first")
    return _SELECTED

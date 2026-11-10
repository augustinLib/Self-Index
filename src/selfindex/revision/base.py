from __future__ import annotations

import operator
from dataclasses import dataclass


@dataclass(frozen=True)
class ProposedKey:
    text: str
    retained_from: int | None = None

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("proposed key must contain text")
        if self.retained_from is not None:
            if isinstance(self.retained_from, bool):
                raise ValueError("retained key id must be an integer")
            object.__setattr__(
                self, "retained_from", operator.index(self.retained_from)
            )

    @property
    def is_new(self):
        return self.retained_from is None

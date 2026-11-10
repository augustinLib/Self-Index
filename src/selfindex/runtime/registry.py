from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Mapping[str, T], Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def validate_name(self, name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.kind}: a nonempty name is required")
        if name in self._items:
            raise ValueError(f"{self.kind} already registered: {name}")

    def register(self, name: str, value: T) -> T:
        self.validate_name(name)
        self._items[name] = value
        return value

    def __getitem__(self, name: str) -> T:
        return self._items[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

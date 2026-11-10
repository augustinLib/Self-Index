import operator
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager


class SnapshotLock:

    def __init__(self):
        self._lock = threading.RLock()
        self._released = threading.Condition(self._lock)
        self._leases = {}

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *error):
        self._lock.release()

    @contextmanager
    def readonly(self):
        thread = threading.get_ident()
        with self:
            self._leases[thread] = self._leases.get(thread, 0) + 1
        try:
            yield
        finally:
            with self:
                self._leases[thread] -= 1
                if not self._leases[thread]:
                    del self._leases[thread]
                self._released.notify_all()

    @contextmanager
    def mutation(self):
        with self:
            if threading.get_ident() in self._leases:
                raise RuntimeError(
                    "cannot mutate an index during a snapshot read lease"
                )
            while self._leases:
                self._released.wait()
            yield


class ScoringView(ABC):

    N: int
    device: object

    @abstractmethod
    def query_repr(self, texts): ...

    @abstractmethod
    def doc_query_repr(self, docs): ...

    def original_scores(self, queries):
        raise NotImplementedError(
            f"{type(self).__name__} does not expose original-document scores"
        )

    @abstractmethod
    def key_scores(self, queries, kids): ...

    @abstractmethod
    def document_scores(self, queries): ...

    @abstractmethod
    def new_key_scores(self, queries, texts): ...

    @property
    @abstractmethod
    def entry_count(self): ...


class LiveIndexView(ScoringView):

    @abstractmethod
    def key_text(self, kid): ...

    @abstractmethod
    def key_owner(self, kid): ...

    @abstractmethod
    def keys_of(self, doc): ...


class KeyIndex(ABC):
    N: int
    accepted_edits: list
    generated_keys: dict[int, list[str]]
    has_embeddings = False

    @abstractmethod
    def add_keys(self, doc, texts): ...

    def replace_validated(self, doc, texts):
        raise NotImplementedError(
            f"{type(self).__name__} does not support exact validated-key replacement"
        )

    def replace_validated_many(self, replacements):
        for doc, keys in replacements.items():
            self.replace_validated(doc, keys)

    def _validated_replacement(self, doc, texts):
        if isinstance(doc, bool):
            raise ValueError("document ID must be an integer, not a boolean")
        doc = operator.index(doc)
        keys = list(texts)
        if not 0 <= doc < self.N or not all(
            isinstance(text, str) and text.strip() for text in keys
        ):
            raise ValueError(
                "exact replacement requires a valid document and nonblank keys"
            )
        return doc, keys

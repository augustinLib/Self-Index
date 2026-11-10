from abc import ABC, abstractmethod


class SearchAdapter(ABC):
    corpus_text: dict

    @abstractmethod
    def search(self, query, k=5): ...

    @abstractmethod
    def search_description(self, k): ...

    def get_document(self, docid):
        text = self.corpus_text.get(str(docid))
        return None if text is None else {"docid": str(docid), "text": text}

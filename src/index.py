import pickle
from abc import ABC, abstractmethod

from src.embedding import Embedder
from src.similarity_measure import Measure
from src.text_splitter import Splitter


class Index:
    def __init__(
        self,
        document: str | None,
        path: str | None,
        embedder: Embedder,
        measure: Measure,
        splitter: Splitter,
    ) -> None:
        super().__init__()

        self.embedder = embedder
        self.measure = measure
        self.splitter = splitter

        if document:
            self.document = document
            self.index = self.index_document()
        elif path:
            self.index = self.load_index(path)
        else:
            raise ValueError("Either document or a path must be provided")

    def index_document(self) -> list[tuple]:
        chunks = self.splitter.split(self.document)
        embeddings = self.embedder.embed(chunks)
        return list(zip(chunks, embeddings))

    def candidates(self, query: str) -> list[str]:
        query_emb = self.embedder.embed([query])[0]
        return self.measure.return_top(query_emb, self.index)

    def save_index(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.index, f)

    def load_index(self, path: str) -> list[tuple]:
        with open(path, "rb") as f:
            return pickle.load(f)

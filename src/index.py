from abc import ABC, abstractmethod

from src.embedding import Embedder
from src.similarity_measure import Measure
from src.text_splitter import Splitter
from src.utils.utils import dump_pickle, read_pickle


class Index:
    def __init__(
        self,
        embedder: Embedder,
        splitter: Splitter,
    ) -> None:
        super().__init__()

        self.splitter = splitter
        self.embedder = embedder

    def __call__(self, document: str, index_path: str | None = None) -> list[str]:
        """returns an indexed document"""
        if document:
            self.index = index = self.index_document(document)
        elif index_path:
            self.index = index = self.load_index(index_path)
        else:
            raise ValueError("Either document or an index_path must be provided")

        return index

    def index_document(self, document: str) -> list[tuple]:
        chunks = self.splitter.split(document)
        embeddings = self.embedder.embed(chunks)
        return list(zip(chunks, embeddings))

    def save_index(self, path: str) -> None:
        assert self.index, "Index is empty"
        dump_pickle(self.index, path=path)

    def load_index(self, path: str) -> list[tuple]:
        return read_pickle(path=path)

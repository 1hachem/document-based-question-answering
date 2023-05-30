from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass


class MiniLM(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


if __name__ == "__main__":
    output = MiniLM().embed(["Hello, my dog is cute", "how to kill a bird"])
    print(output.shape)

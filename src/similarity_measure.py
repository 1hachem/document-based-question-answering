from abc import ABC, abstractmethod

import scipy.spatial as spatial


class Measure(ABC):
    @abstractmethod
    def measure(self, embedding_1: list[float], embedding_2: list[float]) -> float:
        pass

    @abstractmethod
    def return_top(self, query_emb: list[str], index) -> list[str]:
        pass


class CosineSimilarity(Measure):
    def __init__(self, k: int = 3) -> None:
        super().__init__()
        self.top_k = k

    def measure(self, embedding_1: list[float], embedding_2: list[float]) -> float:
        return 1 - spatial.distance.cosine(embedding_1, embedding_2)

    def return_top(self, query_emb: list[str], index: list[tuple]) -> list[str]:
        scores = [self.measure(query_emb, emb) for _, emb in index]
        return [
            index[i][0]
            for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                : self.top_k
            ]
        ]


if __name__ == "__main__":
    vector_1 = [-1.0, -2.0, -3.0]
    vector_2 = [1.0, 2.0, 3.0]
    print(CosineSimilarity().measure(vector_1, vector_2))

from abc import ABC, abstractmethod

import scipy.spatial as spatial


class Measure(ABC):
    @abstractmethod
    def measure(self, embedding_1: list[float], embedding_2: list[float]) -> float:
        pass


class cosine_similarity(Measure):
    def __init__(self) -> None:
        super().__init__()

    def measure(self, embedding_1: list[float], embedding_2: list[float]) -> float:
        return 1 - spatial.distance.cosine(embedding_1, embedding_2)


if __name__ == "__main__":
    vector_1 = [-1.0, -2.0, -3.0]
    vector_2 = [1.0, 2.0, 3.0]
    print("cosine", cosine_similarity().measure(vector_1, vector_2))

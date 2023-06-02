from abc import ABC, abstractmethod

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BertForQuestionAnswering

from src.similarity_measure import CosineSimilarity


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text in a list of texts"""
        pass


class MiniLM(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


class Bert(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        self.model = BertForQuestionAnswering.from_pretrained(
            "deepset/bert-base-cased-squad2"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        inputs = [self.tokenizer(text, return_tensors="pt") for text in texts]
        with torch.no_grad():
            outputs = [
                self.model(**input, output_hidden_states=True) for input in inputs
            ]
            # take the average of the last hidden-state of each token to represent the sentence
            outputs = [
                output.hidden_states[-1].mean(dim=1).flatten().tolist()
                for output in outputs
            ]

        return outputs


if __name__ == "__main__":
    # output = MiniLM().embed(["Hello, my dog is cute", "how to kill a bird"])
    # print(output.shape)

    bert = Bert()
    output = bert.embed(["what is my name", "my name is not bob"])
    print(CosineSimilarity().measure(output[0], output[1]))

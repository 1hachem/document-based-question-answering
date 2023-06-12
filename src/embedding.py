from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, BertForQuestionAnswering

from src.similarity_measure import CosineSimilarity


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of contexts"""
        pass

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of queries"""
        return self.embed(texts)

    def embed_context(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of contexts"""
        return self.embed(texts)


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


class E5(Embedder):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-small")
        self.model = AutoModel.from_pretrained("intfloat/e5-small")

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed(self, texts: list[str]) -> list[list[float]]:
        batch_dict = self.tokenizer(texts, padding=True, return_tensors="pt")

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().tolist()

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.embed(["query :" + text for text in texts])

    def embed_context(self, texts: list[str]) -> list[list[float]]:
        return self.embed(["passage :" + text for text in texts])


if __name__ == "__main__":
    emb = Bert()
    queries = emb.embed_query(["what is my name ?", "what is not my name ?"])
    contexts = emb.embed_context(["my name hachem", "my name is not bob"])
    print(CosineSimilarity().measure(queries[1], contexts[1]))

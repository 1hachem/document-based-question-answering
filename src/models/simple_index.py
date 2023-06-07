from src.embedding import Bert, MiniLM
from src.index import Index
from src.lm import GPT2, GPTJ
from src.similarity_measure import CosineSimilarity
from src.text_splitter import SentenceSplitter, TokenSplitter


class simple_index:
    def __init__(self) -> None:
        self.splitter = SentenceSplitter()
        self.embedder = Bert()
        self.measure = CosineSimilarity(k=2)
        self.llm = GPTJ()

    def __call__(self, context: str, question: str) -> str:
        index = Index(
            embedder=self.embedder,
            splitter=self.splitter,
        )

        indexed_context = index(context)

        query_emb = self.embedder.embed([question])[0]
        candidates = self.measure.return_top(query_emb, indexed_context)

        print(candidates)

        sub_context = "\n\n".join(candidates)

        prompt = f"""from this context : {sub_context}
        answer this question : {question}
        answer :"""

        return self.llm(prompt)

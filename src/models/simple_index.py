from src.index import Index
from src.similarity_measure import CosineSimilarity
from src.text_splitter import SentenceSplitter


class SimpleIndex:
    def __init__(self, embedding, llm) -> None:
        self.splitter = SentenceSplitter()
        self.embedder = embedding()
        self.measure = CosineSimilarity(k=10)
        self.llm = llm()
        self.indexed_context = None

    def index_context(self, context: str):
        index = Index(
            embedder=self.embedder,
            splitter=self.splitter,
        )
        indexed_context = index(context)
        return indexed_context

    def __call__(self, context: str, question: str) -> str:
        query_emb = self.embedder.embed([question])[0]
        if self.indexed_context:
            candidates = self.measure.return_top(query_emb, self.indexed_context)
        else:
            self.indexed_context = self.index_context(context)
            candidates = self.measure.return_top(query_emb, self.indexed_context)

        print(candidates)

        sub_context = "\n\n".join(candidates)

        prompt = f"""from this context : {sub_context}
        answer this question : {question}
        answer :"""

        return self.llm([prompt])[0]

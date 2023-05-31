from src.embedding import MiniLM
from src.index import Index
from src.lm import GPT2
from src.similarity_measure import CosineSimilarity
from src.text_splitter import TokenSplitter


class simple_index:
    def __init__(self) -> None:
        self.splitter = TokenSplitter(chunk_size=20, chunk_overlap=6)
        self.embedder = MiniLM()
        self.measure = CosineSimilarity(k=3)
        self.llm = GPT2()

    def __call__(self, context: str, question: str) -> str:
        index = Index(
            document=context,
            path=None,
            embedder=self.embedder,
            measure=self.measure,
            splitter=self.splitter,
        )

        candidates = index.candidates(question)
        print(candidates)

        sub_context = "\n\n".join(candidates)

        prompt = f"""from this context : {sub_context}
        answer this question : {question}
        answer :"""

        return self.llm(prompt)

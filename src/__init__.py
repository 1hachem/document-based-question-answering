from src.embedding import Bert, MiniLM
from src.lm import GPT2, GPTJ, LlamaQ4, VicunaQ8
from src.similarity_measure import CosineSimilarity

llms = [GPT2, LlamaQ4, VicunaQ8, GPTJ]
embedding_models = [Bert, MiniLM]
measures = [CosineSimilarity]

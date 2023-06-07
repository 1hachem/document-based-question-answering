from src.embedding import Bert, MiniLM
from src.lm import GPT2, GPTJ, LlamaQ4
from src.similarity_measure import CosineSimilarity
from src.text_splitter import (ParagarphSplitter, SentenceSplitter,
                               TokenSplitter)

splitters = [SentenceSplitter, TokenSplitter, ParagarphSplitter]
embedding_models = [Bert, MiniLM]
measures = [CosineSimilarity]
llms = [GPT2, LlamaQ4, GPTJ]

from src.embedding import Bert, MiniLM
from src.lm import GPT2, Falcon
from src.similarity_measure import CosineSimilarity
from src.text_splitter import SentenceSplitter, TokenSplitter
from src.vicuna import Vicuna

splitters = [SentenceSplitter, TokenSplitter]
embedding_models = [Bert, MiniLM]
measures = [CosineSimilarity]
llms = [Vicuna, Falcon]

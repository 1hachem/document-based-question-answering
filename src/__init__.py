from src.embedding import Bert, MiniLM, E5
from src.lm import Falcon, Llama
from src.similarity_measure import CosineSimilarity
from src.text_splitter import SentenceSplitter
from src.vicuna import Vicuna

splitters = [SentenceSplitter]
embedding_models = [Bert, MiniLM, E5]
measures = [CosineSimilarity]
llms = [Vicuna, Falcon, Llama]

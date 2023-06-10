from src.embedding import Bert, MiniLM
from src.lm import GPT2, GPTJ, Falcon, Llama
from src.similarity_measure import CosineSimilarity
from src.text_splitter import ParagarphSplitter, SentenceSplitter
from src.vicuna import Vicuna

splitters = [SentenceSplitter, ParagarphSplitter]
embedding_models = [Bert, MiniLM]
measures = [CosineSimilarity]
llms = [Falcon, Vicuna]

from datasets import load_dataset

from src.benchmark import bench_mark
from src.index import Index
from src.utils.utils import dump_pickle, read_pickle

## Load dataset
dataset = load_dataset("squad", split="validation[:1]")

dataset_val = dataset[:]
dataset_length = len(dataset_val["answers"])

contexts, questions, answers = (
    dataset_val["context"],
    dataset_val["question"],
    [answer["text"][0] for answer in dataset_val["answers"]],
)

print(dataset_length)
if __name__ == "__main__":
    results = bench_mark(answers=answers, contexts=contexts, questions=questions)
    dump_pickle(results, "outputs/squad/test.pkl")
    print(read_pickle("outputs/squad/test.pkl"))

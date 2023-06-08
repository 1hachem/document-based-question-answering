from datasets import load_dataset

from src.benchmark import construct_indexes, inference
from src.utils.utils import save_json

PATH = "outputs/indexes/squad.json"

## Load dataset
dataset = load_dataset("squad", split="validation[:100]")

dataset_val = dataset[:]
dataset_length = len(dataset_val["answers"])

contexts, questions, answers = (
    dataset_val["context"],
    dataset_val["question"],
    [answer["text"][0] for answer in dataset_val["answers"]],
)

print("dataset_length", dataset_length)

if __name__ == "__main__":
    indexes = construct_indexes(answers=answers, contexts=contexts, questions=questions)
    indexes = inference(indexes)
    save_json(indexes, PATH)

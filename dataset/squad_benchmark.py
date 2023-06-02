from datasets import load_dataset

from src.models.simple_index import simple_index
from src.utils.utils import dump_pickle, read_pickle

dataset = load_dataset("squad", split="validation[:2]")

dataset_val = dataset[:]
dataset_length = len(dataset_val["answers"])

index = simple_index()


def bench_mark() -> list[tuple]:
    inference = []
    for context, question, answer in zip(
        dataset_val["context"], dataset_val["question"], dataset_val["answers"]
    ):
        prediction = index(context, question)
        ground_truth = answer["text"][0]

        print("LLM :", prediction)
        print("GT :", ground_truth)
        inference.append((prediction, ground_truth))
    return inference


if __name__ == "__main__":
    results = bench_mark()
    dump_pickle(results, "outputs/squad/test.pkl")
    print(read_pickle("outputs/squad/test.pkl"))

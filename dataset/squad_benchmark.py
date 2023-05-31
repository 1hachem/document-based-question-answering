import asyncio

from datasets import load_dataset

from src.evaluation import is_correct
from src.models.simple_index import simple_index

dataset = load_dataset("squad", split="validation[:10]")

dataset_val = dataset[:]
dataset_length = len(dataset_val["answers"])

index = simple_index()


def bench_mark():
    score = 0
    for context, question, answer in zip(
        dataset_val["context"], dataset_val["question"], dataset_val["answers"]
    ):
        prediction = index(context, question)

        # if await is_correct(prediction, answer["text"][0]):
        # score += 1

        print("LLM :", prediction)
        print("GT :", answer["text"][0])

    return score


if __name__ == "__main__":
    score = bench_mark()
    print(f"score : {score}/{dataset_length}")

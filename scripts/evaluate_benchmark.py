import asyncio

from src.benchmark import evaluate_inferences, parse_results
from src.utils.utils import read_json, save_json

PATH = "results/benchmarks/squad.json"
SAVE_PATH = "results/summary/squad.json"

if __name__ == "__main__":
    indexes = read_json(PATH)
    indexes = asyncio.run(evaluate_inferences(indexes))
    save_json(indexes, PATH)
    table = parse_results(indexes)
    save_json(table, SAVE_PATH)

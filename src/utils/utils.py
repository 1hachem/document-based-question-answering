import json
import pickle


def dump_pickle(somthing: any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(somthing, f)


def read_pickle(path) -> any:
    with open(path, "rb") as f:
        return pickle.load(f)


# Open the JSON file for reading
def read_json(path: str):
    with open(path, "r") as file:
        data = json.load(file)
    return data


def save_json(data: dict, path: str):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

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


def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        list: List of dictionaries parsed from the JSONL file.
    """
    data_list = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data_list.append(json_obj)
    return data_list

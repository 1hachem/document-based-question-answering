import pickle


def dump_pickle(somthing: any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(somthing, f)


def read_pickle(path) -> any:
    with open(path, "rb") as f:
        return pickle.load(f)

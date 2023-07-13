import json
import pathlib
import pickle
from typing import Dict, Tuple, Union

PathType = Union[str, pathlib.Path]


def save_pickle(path_to_file: PathType, obj):
    with open(path_to_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path_to_file: PathType):
    with open(path_to_file, "rb") as f:
        return pickle.load(f)


def save_json(path_to_file: PathType, data: dict):
    with open(path_to_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path_to_file: PathType):
    with open(path_to_file, "rb") as f:
        return json.load(f)

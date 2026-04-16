import json
from typing import List
from .models import FunctionDefinition, PromptInput


def load_json_file(path: str) -> None:

    try:
        with open(path, "r") as file:
            return json.load(file)

    except FileNotFoundError:
        raise Exception(f"File not found: {path}")

    except json.JSONDecodeError:
        raise Exception(f"Invalid JSON: {path}")


def load_functions(path: str) -> List[FunctionDefinition]:
    data = load_json_file(path)
    return [FunctionDefinition(**fn) for fn in data]


def load_prompts(path: str) -> List[PromptInput]:
    data = load_json_file(path)
    return [PromptInput(**p) for p in data]

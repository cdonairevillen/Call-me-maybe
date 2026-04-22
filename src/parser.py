import json
from typing import Any, List
from .models import FunctionDefinition, PromptInput


def load_json_file(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        Exception: If file is missing or invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except FileNotFoundError as exc:
        raise Exception(f"File not found: {path}") from exc

    except json.JSONDecodeError as exc:
        raise Exception(f"Invalid JSON: {path}") from exc


def load_functions(path: str) -> List[FunctionDefinition]:
    """
    Load function definitions from JSON file.

    Args:
        path: Path to definitions file.

    Returns:
        List of validated function definitions.
    """
    data = load_json_file(path)
    functions = []
    for function in data:
        try:
            functions.append(FunctionDefinition(**function))

        except Exception as e:
            print(f"INVALID FUNCTION SKIPPED: {e}")
    return functions


def load_prompts(path: str) -> List[PromptInput]:
    """
    Load prompts from JSON file.

    Args:
        path: Path to prompts file.

    Returns:
        List of validated prompts.
    """
    data = load_json_file(path)
    prompts = []
    for prompt in data:
        try:
            prompts.append(PromptInput(**prompt))

        except Exception as e:
            print(f"INVALID PROMPT SKIPPED: {e}")
    return prompts

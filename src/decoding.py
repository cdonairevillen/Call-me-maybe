import json
from typing import Any


def tensor_to_list(tensor: Any) -> Any:
    """
    Convert a tensor output into a list of token ids.

    Args:
        tensor: Tensor-like object supporting tolist().

    Returns:
        List of token ids.
    """
    return tensor.tolist()[0]


def load_vocab(path: str) -> dict[int, str]:
    """
    Load a vocabulary JSON file and reverse key/value mapping.

    Args:
        path: Path to vocabulary JSON file.

    Returns:
        Dictionary mapping token id to token string.
    """
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return {value: key for key, value in vocab.items()}


def token_id_to_str(vocab: dict[str, int], token_id: int) -> str:
    """
    Convert a token id into its string representation.

    Args:
        vocab: Vocabulary dictionary.
        token_id: Token id to search.

    Returns:
        Matching token string.

    Raises:
        ValueError: If token id is not found.
    """
    for token, value in vocab.items():
        if value == token_id:
            return token

    raise ValueError("Token not found.")

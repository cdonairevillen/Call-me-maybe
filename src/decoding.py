import json


def tensor_to_list(tensor):
    return tensor.tolist()[0]


def load_vocab(path: str):
    with open(path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    return {v: k for k, v in vocab.items()}


def token_id_to_str(vocab: dict, token_id: int) -> str:
    for k, v in vocab.items():
        if v == token_id:
            return k
    raise ValueError("TOKEN NOT FOUND")

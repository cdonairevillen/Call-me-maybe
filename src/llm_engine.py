from llm_sdk import Small_LLM_Model
from .decoding import tensor_to_list, load_vocab
import re
import json
from typing import Any


class LLMEngine():

    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[int, str] = load_vocab(
            self.model.get_path_to_vocab_file()
        )
        # reverse_vocab: {token_str (str): token_id (int)}
        self.reverse_vocab: dict[str, int] = {
            v: k for k, v in self.vocab.items()
        }

        self.POSITIVE_RULES: dict[str, list[str]] = {
            "fn_add_numbers": ["sum", "add", "plus", "+", "increase"],
            "fn_greet": ["greet", "hello", "hi", "name"],
            "fn_reverse_string": ["reverse", "backwards", "flip"],
            "fn_get_square_root": ["square root", "sqrt", "root"],
            "fn_substitute_string_with_regex": [
                "replace", "substitute", "regex", "change", "with", "word"
            ],
        }

        self.NEGATIVE_RULES: dict[str, list[str]] = {
            "fn_add_numbers": [
                "multiply", "times", "product", "divide"],
            "fn_greet": ["reverse", "sum", "square", "root", "replace",
                         "substitute", "calculate"],
            "fn_reverse_string": ["sum", "add", "square", "root"],
            "fn_get_square_root": ["sum", "add", "greet", "replace"],
            "fn_substitute_string_with_regex": [
                "add", "sum", "square", "root"
            ],
        }

    def encode(self, text: str) -> list[int]:
        return tensor_to_list(self.model.encode(text))

    def logits(self, tokens: list[int]) -> list[float]:
        return self.model.get_logits_from_input_ids(tokens)

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token ids back to a string."""
        return "".join(self.vocab.get(tid, "") for tid in token_ids)

    def score_function(self, prompt_l: str, fn: Any) -> float:
        score: float = 0.0

        for kw in self.POSITIVE_RULES.get(fn.name, []):
            if kw in prompt_l:
                score += 20.0

        for kw in self.NEGATIVE_RULES.get(fn.name, []):
            if kw in prompt_l:
                score -= 30.0

        return score

    @staticmethod
    def build_function_selection_prompt(
        prompt: str, functions: list[Any]
    ) -> str:
        functions_text = ""

        for fn in functions:
            params = ", ".join(
                [f"{k}:{v.type}" for k, v in fn.parameters.items()]
            )
            functions_text += (
                f"- {fn.name}({params})\n"
                f"  description: {fn.description}\n\n"
            )

        return f"""
        You are a STRICT function routing system.

        Your job is to select exactly ONE function that best matches the
        USER REQUEST.

        You must follow these PRIORITY RULES (VERY IMPORTANT):

        1. MATH OPERATIONS HAVE HIGHEST PRIORITY
        If the request contains ANY arithmetic question, calculation, numbers,
        or math operators (+, -, *, /, "sum", "plus", "times"), you MUST select
        a math function.
        If you find numeric parameters w

        2. STRING OPERATIONS ARE SECOND PRIORITY
        Only choose string functions if the request is clearly about text
        manipulation.

        3. GREETINGS ARE LOWEST PRIORITY
        Greeting words like "hello", "hi", "name" are NEVER enough to select a
        greet function if ANY other task is present.

        4. IGNORE conversational noise
        Phrases like:
        - "hello, I have a question"
        - "I need help"
        - "can you"
        are NOT relevant for function selection.

        5. If multiple functions seem valid, ALWAYS prefer:
        math > string > greet

        6. If no function clearly matches, return "Function not found"

        ---

        AVAILABLE FUNCTIONS:
        {functions_text}

        ---

        OUTPUT FORMAT (STRICT JSON ONLY):
        {{
        "name": "<function_name or Function not found>",
        "parameters": {{}}
        }}

        USER REQUEST:
        {prompt}

        ANSWER:
    """

    @staticmethod
    def build_parameter_extraction_prompt(prompt: str, fn: Any) -> str:
        params_desc = "\n".join(
            [
                f"- {k} ({v.type}): {getattr(v, 'description', '')}"
                for k, v in fn.parameters.items()
            ]
        )
        param_keys = ", ".join(
            [f'"{k}": <value>' for k in fn.parameters]
        )

        return f"""
        Extract the parameter values for the function "{fn.name}"
        from the user request.

        IMPORTANT RULES:
        - Use ONLY information from the user request.
        - Convert number words into integers (two -> 2, five -> 5).
        - If multiple numbers exist, preserve order.
        - Do NOT guess missing values.
        - Output ONLY valid JSON.

        PARAMETERS NEEDED:
        {params_desc}

        USER REQUEST: "{prompt}"

        OUTPUT FORMAT (STRICT JSON ONLY, no extra text):
        {{{param_keys}}}

        ANSWER:
        """

    def score_token_sequence(
                            self,
                            initial_logits: list[float],
                            fn_tokens: list[int],
                            input_ids: list[int],
                            ) -> float:
        score: float = 0.0
        current_ids: list[int] = input_ids[:]
        current_logits: list[float] = initial_logits

        for token_id in fn_tokens:
            score += current_logits[token_id]
            current_ids.append(token_id)
            current_logits = self.logits(current_ids)

        return score

    def generate_tokens(self, input_ids: list[int],
                        max_tokens: int = 60) -> list[int]:
        generated: list[int] = []
        eos_id: int = self.reverse_vocab.get("<eos>", -1)

        for _ in range(max_tokens):
            logits = self.logits(input_ids + generated)

            top_k = sorted(range(len(logits)),
                           key=lambda i: logits[i], reverse=True)[:5]

            next_token_id = top_k[0]  # try random.choice(top_k) if skip heuristic

            if next_token_id in (eos_id, -1):
                break

            generated.append(next_token_id)

            text = self.decode(generated)
            if "}" in text:
                break

        return generated

    def fallback_heuristic(self, prompt: str,
                           functions: list[Any]) -> dict[str, Any]:
        prompt_l = prompt.lower()

        best_fn = None
        best_score = float("-inf")

        for fn in functions:
            score = self.score_function(prompt_l, fn)

            if score > best_score:
                best_score = score
                best_fn = fn

        if best_fn is None or best_score < 10:
            return {
                "prompt": prompt,
                "name": "Function not found",
                "parameters": {}}

        return {
            "prompt": prompt,
            "name": best_fn.name,
            "parameters": self.extract_parameters(prompt, best_fn),
        }

    @staticmethod
    def parse_json_output(text: str) -> dict[str, Any]:
        """Extract and parse the first JSON object found in text."""
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {}

    def extract_parameters_heuristic(
        self, prompt: str, fn: Any
    ) -> dict[str, Any]:

        """Fallback parameter extraction using regex."""
        params: dict[str, Any] = {}
        numbers = re.findall(r"\d+\.?\d*", prompt)
        str_match = re.search(r"['\"](.+?)['\"]", prompt)

        for i, (key, val) in enumerate(fn.parameters.items()):
            if val.type in ("int", "float", "number") and i < len(numbers):
                raw = numbers[i]
                params[key] = float(raw) if "." in raw else int(raw)
            elif val.type == "str":
                params[key] = str_match.group(1) if str_match else ""

        return params

    def extract_parameters(self, prompt: str, fn: Any) -> dict[str, Any]:
        if not fn.parameters:
            return {}

        extraction_prompt = self.build_parameter_extraction_prompt(prompt, fn)
        input_ids = self.encode(extraction_prompt)
        generated = self.generate_tokens(input_ids)
        output_text = self.decode(generated)

        parsed = self.parse_json_output(output_text)
        if parsed:
            return parsed

        return self.extract_parameters_heuristic(prompt, fn)

    def clean_prompt(self, prompt: str) -> str:
        prompt = prompt.lower()

        noise_patterns = [
            r"hello.*?,",
            r"hi.*?,",
            r"i have a question about.*?,",
            r"can you",
            r"please",
        ]

        for p in noise_patterns:
            prompt = re.sub(p, "", prompt)

        return prompt.strip()

    def generate_function_call(self, prompt: str,
                               functions: list[Any]) -> dict[str, Any]:

        # LLM Setter
        full_prompt = self.build_function_selection_prompt(prompt, functions)

        input_ids = self.encode(full_prompt)

        # Token Generator
        generated_tokens = self.generate_tokens(input_ids, max_tokens=60)

        output_text = self.decode(generated_tokens)

        parsed = self.parse_json_output(output_text)

        # LLM choose
        if parsed and "name" in parsed:
            fn_name = parsed["name"]

            if fn_name == "Function not found":
                return parsed

            for fn in functions:
                if fn.name == fn_name:

                    clean_prompt = self.clean_prompt(prompt)
                    return {
                        "prompt": prompt,
                        "name": fn.name,
                        "parameters": self.extract_parameters(clean_prompt, fn)
                    }

        # FALLBACK
        return self.fallback_heuristic(prompt, functions)

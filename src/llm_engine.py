from llm_sdk import Small_LLM_Model
from .decoding import tensor_to_list, load_vocab
import re
import numpy as np
from typing import Any

NUM_RE = re.compile(r"-?\d+\.?\d*")
ALPHA_RE = re.compile(r"[a-zA-Z]")
LOWER_WORDS_RE = re.compile(r"[a-z]+")
QUOTED_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")


class LLMEngine:
    """Route natural language prompts to function calls.

    Uses a small language model for function selection via constrained
    logit scoring, and deterministic regex-based extraction for
    parameter values. This avoids relying on the model to spontaneously
    produce structured output.
    """

    WORD_TO_NUM: dict[str, int] = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12,
    }

    IGNORED_WORDS: frozenset[str] = frozenset({
        "what", "is", "the", "of", "and", "please",
        "can", "you", "i", "have", "a", "question", "about",
    })

    NUMERIC_TYPES: frozenset[str] = frozenset(
        {"number", "int", "float", "integer"}
    )

    STRING_TYPES: frozenset[str] = frozenset({"string", "str"})

    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.vocab: dict[int, str] = load_vocab(
            self.model.get_path_to_vocab_file()
        )
        self.reverse_vocab: dict[str, int] = {
            v: k for k, v in self.vocab.items()
        }
        self.encode_cache: dict[str, list[int]] = {}
        self.logits_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Core model interface
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids, caching results."""
        cached = self.encode_cache.get(text)
        if cached is not None:
            return cached
        tokens: list[int] = tensor_to_list(self.model.encode(text))
        self.encode_cache[text] = tokens
        return tokens

    def logits(self, tokens: list[int]) -> np.ndarray:
        """Return the logit vector for the next token given a token sequence.

        Results are cached by a hash of the token sequence to avoid
        redundant forward passes.
        """
        key = hash(tuple(tokens))
        cached = self.logits_cache.get(key)
        if cached is not None:
            return cached
        result = np.asarray(
            self.model.get_logits_from_input_ids(tokens),
            dtype=np.float32,
        )
        self.logits_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def is_numeric(self, type_str: str) -> bool:
        """Return True if type_str represents a numeric parameter type."""
        return type_str.lower() in self.NUMERIC_TYPES

    def is_string(self, type_str: str) -> bool:
        """Return True if type_str represents a string parameter type."""
        return type_str.lower() in self.STRING_TYPES

    # ------------------------------------------------------------------
    # Function classification
    # ------------------------------------------------------------------

    def is_substitution_fn(self, fn: Any) -> bool:
        """Return True if fn is a text-substitution function.

        A function is classified as substitution when it has at least two
        string parameters and its name or description contains a
        replacement-related keyword (replace, substitute, regex, swap).
        Parameter names are intentionally not checked so the classifier
        generalises to any naming convention.
        """
        string_param_count = sum(
            1 for p in fn.parameters.values()
            if self.is_string(p.type)
        )
        name_hints = re.search(
            r"\b(replace|substitute|regex|swap)\b",
            fn.name + " " + fn.description,
            re.IGNORECASE,
        )
        return string_param_count >= 2 and name_hints is not None

    # ------------------------------------------------------------------
    # Function selection
    # ------------------------------------------------------------------

    def build_selection_prompt(
        self,
        prompt: str,
        functions: list[Any],
    ) -> str:
        """Build the context prompt used for logit-based function scoring.

        The prompt ends just before the model would write the function
        name, so the logit vector at that position directly encodes the
        model's belief about which function is correct.
        """
        functions_text = "".join(
            f"- {fn.name}("
            + ", ".join(
                f"{k}:{v.type}" for k, v in fn.parameters.items()
            )
            + f"): {fn.description}\n"
            for fn in functions
        )
        return (
            "        Pick the ONE function that matches the request.\n"
            "        FUNCTIONS:\n"
            f"        {functions_text}"
            "        - Function not found: operation not supported\n"
            "        Rules:\n"
            "        - Match by action verb and operand type\n"
            "        - Numbers -> numeric function\n"
            "        - Names or words -> string function\n"
            '        - "replace", "substitute" + quoted text'
            " -> ALWAYS string function,\n"
            "          even if the text contains numbers\n"
            "        - If the action verb is missing or unclear,"
            " choose Function not found\n"
            "        - Never invent unsupported operations\n"
            f"        REQUEST: {prompt}\n"
            "        ANSWER:\n"
            "        "
        )

    def sequence_logprob(
        self,
        prefix_ids: list[int],
        tokens: list[int],
    ) -> float:
        """Compute the mean log-probability of a token sequence.

        Performs one forward pass per token in the sequence beyond the
        shared prefix, reading the log-probability of each token from
        the model's logit distribution at that position.
        """
        score = 0.0
        current_ids = prefix_ids[:]
        current_logits = self.logits(current_ids)
        for i, token_id in enumerate(tokens):
            score += float(current_logits[token_id])
            if i < len(tokens) - 1:
                current_ids = current_ids + [token_id]
                current_logits = self.logits(current_ids)
        return score / len(tokens)

    def rank_substitution_functions(
        self,
        prompt: str,
        substitution_fns: list[Any],
    ) -> Any:
        """Rank substitution function candidates using logit scoring."""
        sel_prompt = self.build_selection_prompt(
            prompt, substitution_fns
        )
        base_prompt = self.build_selection_prompt(
            "...", substitution_fns
        )
        input_ids = self.encode(sel_prompt)
        baseline_ids = self.encode(base_prompt)
        best_fn: Any = substitution_fns[0]
        best_score: float = float("-inf")
        for fn in substitution_fns:
            tokens = self.encode(fn.name)
            score = (
                self.sequence_logprob(input_ids, tokens)
                - self.sequence_logprob(baseline_ids, tokens)
            )
            if score > best_score:
                best_score = score
                best_fn = fn
        return best_fn

    def select_function(
        self,
        prompt: str,
        functions: list[Any],
    ) -> Any | None:
        """Select the best matching function for a prompt.

        Selection proceeds through three stages:

        1. Hard overrides — fast deterministic checks that short-circuit
           the logit scoring for unambiguous cases (empty prompt,
           replace/substitute verbs with quoted text).

        2. Keyword filter — rejects prompts whose meaningful words have
           no overlap with any word derived from the available function
           names and descriptions, avoiding spurious logit matches.

        3. Logit ranking — scores each function name by its mean
           log-probability in the context of the selection prompt minus
           a baseline, using constrained decoding rather than free
           text generation.
        """
        NOT_FOUND_MARGIN = 0.8

        lowered = prompt.lower()

        if not ALPHA_RE.search(prompt):
            return None

        has_replace_verb = bool(
            re.search(r"\b(replace|substitute)\b", lowered)
        )
        has_quoted_text = bool(re.search(r"['\"]", prompt))
        has_in_clause = bool(re.search(r"\bin\b", lowered))

        if has_replace_verb and (has_quoted_text or has_in_clause):
            substitution_fns = [
                fn for fn in functions if self.is_substitution_fn(fn)
            ]
            if substitution_fns:
                if len(substitution_fns) == 1:
                    return substitution_fns[0]
                return self.rank_substitution_functions(
                    prompt, substitution_fns
                )

        supported_words: set[str] = set()
        for fn in functions:
            source = (
                fn.name.replace("_", " ") + " " + fn.description
            ).lower()
            for word in LOWER_WORDS_RE.findall(source):
                if len(word) > 2:
                    supported_words.add(word)

        prompt_keywords = [
            w for w in LOWER_WORDS_RE.findall(lowered)
            if w not in self.IGNORED_WORDS
        ]

        if len(prompt_keywords) == 1:
            if prompt_keywords[0] not in supported_words:
                return None
        elif prompt_keywords:
            if not any(w in supported_words for w in prompt_keywords):
                return None

        sel_prompt = self.build_selection_prompt(prompt, functions)
        base_prompt = self.build_selection_prompt("...", functions)
        input_ids = self.encode(sel_prompt)
        baseline_ids = self.encode(base_prompt)

        candidates: list[tuple[Any | None, list[int]]] = [
            (fn, self.encode(fn.name)) for fn in functions
        ]
        candidates.append((None, self.encode("Function not found")))

        scored: list[tuple[Any | None, float]] = []
        for obj, tokens in candidates:
            if not tokens:
                continue
            score = (
                self.sequence_logprob(input_ids, tokens)
                - self.sequence_logprob(baseline_ids, tokens)
            )
            scored.append((obj, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_obj, best_score = scored[0]

        if best_obj is None and len(scored) > 1:
            second_obj, second_score = scored[1]
            if (
                second_obj is not None
                and (best_score - second_score) < NOT_FOUND_MARGIN
            ):
                return second_obj

        if best_obj is not None and len(prompt.strip().split()) <= 1:
            return None

        return None if best_obj is None else best_obj

    # ------------------------------------------------------------------
    # Value extraction helpers
    # ------------------------------------------------------------------

    def extract_numbers(self, prompt: str) -> list[int | float]:
        """Extract all numeric values from a prompt.

        Handles integers, decimals, and negative numbers expressed as
        digits (e.g. -5, 3.14) as well as English number words up to
        twelve. Digit values are returned first in left-to-right order,
        followed by any word-form values found.
        """
        found: list[int | float] = [
            float(r) if "." in r else int(r)
            for r in NUM_RE.findall(prompt)
        ]
        lowered = prompt.lower()
        for word, value in self.WORD_TO_NUM.items():
            if re.search(r"\b" + word + r"\b", lowered):
                found.append(value)
        return found

    def extract_quoted_strings(self, prompt: str) -> list[str]:
        """Return all strings found between single or double quotes."""
        return [
            a if a else b
            for a, b in QUOTED_RE.findall(prompt)
        ]

    def extract_number_param(
        self,
        prompt: str,
        already_extracted: dict[str, Any],
    ) -> int | float:
        """Extract the next numeric value not yet assigned to a parameter.

        Uses the count of already-extracted parameters as an index into
        the ordered list of numbers found in the prompt.
        """
        nums = self.extract_numbers(prompt)
        idx = len(already_extracted)
        return nums[idx] if idx < len(nums) else 0

    def extract_replacement(self, prompt: str) -> str:
        """Extract the replacement value from a substitution prompt.

        Priority order:
        1. Quoted string immediately following the word 'with'.
        2. Known symbol words (asterisk, underscore, dash, slash).
        3. First word or token after 'with', including ALL_CAPS tokens.
        4. Default value 'other' when no replacement is specified.
        """
        low = prompt.lower()

        with_quoted = re.search(
            r"\bwith\b\s*['\"]([^'\"]+)['\"]",
            prompt,
            re.IGNORECASE,
        )
        if with_quoted:
            return with_quoted.group(1)

        symbol_map = {
            "asterisk": "*",
            "underscore": "_",
            "dash": "-",
            "slash": "/",
        }
        for word, symbol in symbol_map.items():
            if word in low:
                return symbol

        with_word = re.search(
            r"\bwith\s+([A-Za-z0-9_*]+)",
            prompt,
            re.IGNORECASE,
        )
        if with_word:
            return with_word.group(1)

        return "other"

    def infer_pattern(self, prompt: str) -> str | None:
        """Infer a regex character class from semantic keywords.

        Returns a regex string when the prompt describes a character
        category (vowels, consonants, digits) rather than a literal
        pattern. Returns None if no semantic category is detected.
        """
        low = prompt.lower()
        if "vowel" in low:
            return "[aeiouAEIOU]"
        if "consonant" in low:
            return (
                "[bcdfghjklmnpqrstvwxyz"
                "BCDFGHJKLMNPQRSTVWXYZ]"
            )
        if "number" in low or "digit" in low:
            return "[0-9]"
        return None

    def extract_substitution_params(self, prompt: str,
                                    fn: Any) -> dict[str, Any]:
        """Extract parameters for any text-substitution function.

        Maps the three logical roles of a substitution operation to the
        actual parameter names of the function by position, so the
        extractor works regardless of how the parameters are named:

        - Role 0 (source): the text to be modified — last quoted string,
          or empty string if none is present.
        - Role 1 (pattern): what to search for — semantic regex if the
          prompt describes a character class, otherwise the first quoted
          string.
        - Role 2 (replacement): what to replace with — extracted by
          extract_replacement.

        Any additional parameters beyond the first three receive an
        empty string.
        """
        quoted = self.extract_quoted_strings(prompt)

        source = quoted[-1] if quoted else ""
        pattern = self.infer_pattern(prompt) or (
            quoted[0] if quoted else ""
        )
        replacement = self.extract_replacement(prompt)

        role_values: list[Any] = [source, pattern, replacement]
        param_names = list(fn.parameters.keys())

        return {
            name: role_values[i] if i < len(role_values) else ""
            for i, name in enumerate(param_names)
        }

    def extract_string_param(self, prompt: str,
                             param_name: str) -> str:
        """Extract a single string parameter by its conventional name.

        Handles well-known parameter names (name, s, word, source_string,
        regex, replacement) with targeted extraction logic. Falls back to
        the first quoted string for unrecognised parameter names.
        """
        quoted = self.extract_quoted_strings(prompt)

        if param_name == "name":
            words = prompt.strip().split()
            return words[-1] if len(words) > 1 else ""

        if param_name in ("s", "word"):
            return quoted[0] if quoted else ""

        if param_name == "source_string":
            return quoted[-1] if quoted else ""

        if param_name == "regex":
            return (
                self.infer_pattern(prompt)
                or (quoted[0] if quoted else ".")
            )

        if param_name == "replacement":
            return self.extract_replacement(prompt)

        return quoted[0] if quoted else ""

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    def param_selector(self, prompt: str, param_name: str,
                       param_type: str,
                       already_extracted: dict[str, Any]) -> Any:
        """Extract a single parameter value given its name and type."""
        if self.is_numeric(param_type):
            return self.extract_number_param(
                prompt, already_extracted
            )
        if self.is_string(param_type):
            return self.extract_string_param(prompt, param_name)
        return ""

    def extract_parameters(self, prompt: str,
                           fn: Any) -> dict[str, Any]:
        """Extract all parameters for a matched function.

        Substitution functions use the generic role-based extractor.
        All other functions extract parameters one at a time using their
        declared names and types.
        """
        if not fn.parameters:
            return {}

        if self.is_substitution_fn(fn):
            return self.extract_substitution_params(prompt, fn)

        params: dict[str, Any] = {}
        for param_name, param_def in fn.parameters.items():
            params[param_name] = self.param_selector(
                prompt, param_name, param_def.type, params
            )
        return params

    # MAIN FUNCTION
    def generate_function_call(self, prompt: str,
                               functions: list[Any]) -> dict[str, Any]:
        """Map a natural language prompt to a function call.

        Returns a dict with keys 'prompt', 'name', and 'parameters'.
        If no function matches, 'name' is 'Function not found' and
        'parameters' is an empty dict.
        """
        fn = self.select_function(prompt, functions)

        if fn is None:
            return {
                "prompt": prompt,
                "name": "Function not found",
                "parameters": {},
            }

        return {
            "prompt": prompt,
            "name": fn.name,
            "parameters": self.extract_parameters(prompt, fn),
        }

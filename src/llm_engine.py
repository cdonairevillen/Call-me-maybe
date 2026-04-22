import re
from typing import Any

import numpy as np
from llm_sdk.__init__ import Small_LLM_Model

from .decoding import load_vocab, tensor_to_list

NUM_RE = re.compile(r"(?<![a-zA-Z])-?\d+\.?\d*")
LOWER_WORDS_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")
QUOTED_RE = re.compile(r"'([^']*)'|\"([^\"]*)\"")

REPLACEMENT_STOPWORDS: frozenset[str] = frozenset({
    "with", "by", "using", "via", "for", "from", "into", "as",
    "con", "por", "mediante",
})

INSTRUCTION_VERBS: frozenset[str] = frozenset({
    "replace", "substitute", "change", "reverse", "greet",
    "compute", "calculate", "reemplaza", "sustituye",
})


class LLMEngine:
    """Route natural language prompts into structured function calls.

    Function selection uses constrained logit ranking against a small
    local model. Parameter extraction combines deterministic guards with
    logit-based candidate ranking. Hardcoded rules are kept minimal and
    are only added when they measurably improve reliability.
    """

    WORD_TO_NUM: dict[str, int] = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12,
    }

    NUMERIC_TYPES: frozenset[str] = frozenset(
        {"number", "int", "float", "integer"}
    )

    STRING_TYPES: frozenset[str] = frozenset({"string", "str", "text"})

    IGNORED_WORDS: frozenset[str] = frozenset({
        "what", "is", "the", "of", "and", "please",
        "can", "you", "a", "an", "to", "for",
        "calculate", "compute", "give", "show",
        "tell", "me", "about", "would",
    })

    CONCEPT_MAP: dict[str, str] = {
        "vowels": "[aeiouAEIOU]",
        "vocales": "[aeiouAEIOU]",
        "vowel": "[aeiouAEIOU]",
        "consonants": "[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]",
        "consonant": "[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]",
        "numbers": "[0-9]",
        "number": "[0-9]",
        "digits": "[0-9]",
        "digit": "[0-9]",
        "letters": "[A-Za-z]",
        "letter": "[A-Za-z]",
        "NUMBERS": "NUMBERS",
        "numbers_word": "NUMBERS",
        "spaces": " ",
        "space": " ",
        "asterisk": "*",
        "asterisks": "*",
        "star": "*",
        "stars": "*",
        "underscore": "_",
        "low bar": "_",
        "dash": "-",
        "hyphen": "-",
        "slash": "/",
        "bar": "|",
        "side bar": "|",
    }

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

    # Model Interface

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

    def sequence_logprob(self, prefix_ids: list[int],
                         tokens: list[int]) -> float:
        """Compute the mean log-probability of a token sequence.

        Performs one forward pass per token beyond the shared prefix,
        reading the log-probability of each token from the model's logit
        distribution at that position. Returns negative infinity for
        empty sequences.
        """
        if not tokens:
            return float("-inf")
        score = 0.0
        ids = prefix_ids[:]
        for i, token in enumerate(tokens):
            raw_logits = self.logits(ids)
            score += float(raw_logits[token])
            if i < len(tokens) - 1:
                ids = ids + [token]
        return score / len(tokens)

    # Type Helpers

    def is_numeric(self, type_str: str) -> bool:
        """Return True if type_str represents a numeric parameter type."""
        return type_str.lower() in self.NUMERIC_TYPES

    def is_string(self, type_str: str) -> bool:
        """Return True if type_str represents a string parameter type."""
        return type_str.lower() in self.STRING_TYPES

    def normalize_text(self, text: str) -> str:
        """Resolve a concept keyword to its regex or symbol equivalent."""
        return self.CONCEPT_MAP.get(text.lower().strip(), text)

    def prompt_is_corrupted(self, prompt: str) -> bool:
        """Return True if the prompt should be rejected before processing.

        Rejects empty prompts, prompts with excessive non-ASCII symbols,
        and prompts with unmatched quote characters.
        """
        if not prompt.strip():
            return True
        weird = sum(1 for ch in prompt if ord(ch) > 255)
        if weird > max(2, len(prompt) // 6):
            return True
        remaining = re.sub(QUOTED_RE, "", prompt)
        if "'" in remaining or '"' in remaining:
            return True
        return False

    def is_invalid_string(self, text: str) -> bool:
        """Return True if text is too corrupted to be a valid candidate.

        Rejects empty strings, strings with excessive non-ASCII bytes,
        and strings where fewer than 20 % of characters are alphabetic.
        """
        if not text or not text.strip():
            return True
        weird = sum(1 for ch in text if ord(ch) > 255)
        alpha = sum(1 for ch in text if ch.isalpha())
        if len(text) > 0 and alpha / len(text) < 0.2:
            return True
        return weird > max(1, len(text) // 6)

    # Function Selection

    def build_selection_prompt(self, prompt: str,
                               functions: list[Any]) -> str:
        """Build the context prompt used for logit-based function scoring.

        The prompt ends just before the model would write the function
        name, so the logit vector at that position directly encodes the
        model's belief about which function is correct.
        """
        body = "".join(
            f"- {fn.name}("
            + ", ".join(
                f"{k}:{v.type}" for k, v in fn.parameters.items()
            )
            + f"): {fn.description}\n"
            for fn in functions
        )
        return (
            "Choose EXACTLY ONE function name.\n"
            "Be literal. Be strict. Be conservative.\n"
            "If request is incomplete, unclear, unsupported or missing"
            " required values -> choose Function not found.\n\n"
            "NEVER hallucinate closest function.\n"
            "NEVER convert minus into add.\n"
            "NEVER convert divide into multiply.\n"
            "NEVER convert replace into math.\n"
            "NEVER force a match.\n\n"
            "PRIORITY HIERARCHY (apply strictly in this order):\n"
            "  1. NUMERIC functions: if the prompt contains numbers, numeric"
            " words, or arithmetic operators -> prefer numeric functions"
            " first."
            " A greeting word at the start does NOT override this.\n"
            "  2. STRING-TRANSFORM functions: if the prompt requests text"
            " manipulation (replace, substitute, reverse, regex) -> prefer"
            " these over greeting functions.\n"
            "  3. GREETING functions: ONLY if the prompt's sole and complete"
            " intent is to greet a named person, with no other operation.\n\n"
            "INTENT MAP:\n"
            "- add = add sum plus total combine + más mas\n"
            "- subtract = minus subtract less difference take away - menos\n"
            "- multiply = multiply times product x * por veces\n"
            "- square root = root sqrt radical\n"
            "- reverse = reverse backwards invert string text\n"
            "- greet = ONLY IF the sole intent is to greet a named person."
            " hello/hi at the START of a math or other question"
            " is NOT greet.\n"
            "- regex replace = replace substitute change pattern regex"
            " vowels consonants numbers digits letters\n\n"
            "STRICT EXAMPLES:\n"
            "What is 2 plus 3 -> add\n"
            "What is 2 + 3 -> add\n"
            "What is 2 + two -> add\n"
            "hello, what is 2 + two -> add  (hello is context, not intent)\n"
            "I have a question, what is 2 + two -> add\n"
            "What is 2 minus 3 -> subtract only if subtract exists\n"
            "What is 2 minus 3 -> Function not found if subtract absent\n"
            "Replace vowels in hello with * -> regex replace\n"
            "Replace all numbers in X with NUMBERS -> regex replace\n"
            "Replace all consonants in 'text' with * -> regex replace\n"
            "Substitute 'rat' with 'dog' in 'the rat sat' -> regex replace\n"
            "replace everything that looks like a vowel -> regex replace\n"
            "Greet shrek -> greet\n"
            "Greet john -> greet\n"
            "Root -> Function not found\n"
            "add -> Function not found\n"
            "multiply -> Function not found\n\n"
            "SEMANTIC RULES (HARD):\n"
            "- greet ONLY fires when the prompt's sole purpose is greeting"
            " a named person. A greeting word followed by a math expression"
            " or question -> choose the math function, NOT greet.\n\n"
            "- If user requests transformation WITHOUT explicit source text"
            " -> source_string MUST be \"\"\n\n"
            "- If concept-based replacement is used:\n"
            "examples:\n"
            "    vowels -> [aeiouAEIOU]\n"
            "    consonants -> [b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z]\n"
            "    numbers -> [0-9]\n"
            "    digits -> [0-9]\n"
            "    asterisks -> \"*\"\n"
            "    low bar -> \"_\"\n"
            "    side bar -> \"|\"\n\n"
            "- If prompt is unclear or abstract:\n"
            "NEVER guess string content -> choose Function not found\n\n"
            "- If prompt is nonsense, ASCII-invalid or corrupted:\n"
            "-> Function not found\n\n"
            "FUNCTIONS:\n"
            f"{body}"
            "- Function not found\n\n"
            f"REQUEST: {prompt}\n"
            "ANSWER:\n"
        )

    def keyword_overlap(self, prompt: str, fn: Any) -> int:
        """Return the number of meaningful words shared by prompt and fn."""
        prompt_words = {
            w.lower()
            for w in LOWER_WORDS_RE.findall(prompt)
            if w.lower() not in self.IGNORED_WORDS
        }
        fn_words = {
            w.lower()
            for w in LOWER_WORDS_RE.findall(fn.name + " " + fn.description)
            if len(w) > 2
        }
        return len(prompt_words & fn_words)

    def select_function(self, prompt: str,
                        functions: list[Any]) -> Any | None:
        """Select the best matching function for a prompt.

        Applies a cascade of deterministic guards before logit ranking.
        Guards reject clearly invalid prompts and enforce semantic
        constraints that the small model cannot reliably handle on its
        own. The logit ranker scores remaining candidates by their mean
        log-probability relative to a neutral baseline.
        """
        if self.prompt_is_corrupted(prompt):
            return None

        low = prompt.lower()
        names = {fn.name.lower() for fn in functions}

        only_numbers = re.compile(r"^[\d\s\.\-,]+$")
        if only_numbers.match(prompt.strip()):
            return None

        greet_only = re.compile(
            r"^(greet|hello|hi|hola|saluda|hey)\s*\w*\s*$",
            re.IGNORECASE,
        )
        has_greet_fn = any(
            any(
                w in fn.name.lower() or w in fn.description.lower()
                for w in ["greet", "salud", "welcome", "hello"]
            )
            for fn in functions
        )
        if greet_only.match(prompt.strip()) and not has_greet_fn:
            return None

        minus_as_negative = re.compile(
            r"\bminus\s+(\d+|zero|one|two|three|four|five"
            r"|six|seven|eight|nine|ten|eleven|twelve)\b",
            re.IGNORECASE,
        )
        if any(w in low for w in ["minus", "subtract", "less"]):
            if not any("subtract" in n for n in names):
                if not minus_as_negative.search(prompt):
                    return None

        math_signals = re.compile(
            r"(\d+\s*[+\-*/]\s*(\d+|\w+))"
            r"|(\b(sum|plus|minus|times|multiply|sqrt|root|radical"
            r"|divided|product|veces|más|mas|menos)\b)",
            re.IGNORECASE,
        )
        if math_signals.search(prompt):
            numeric_fns = [
                fn for fn in functions
                if not all(
                    self.is_string(p.type)
                    for p in fn.parameters.values()
                )
            ]
            if numeric_fns:
                functions = numeric_fns

        filtered = functions[:]
        if any(w in low for w in ["replace", "substitute", "change"]):
            regex_fns = [
                fn for fn in filtered
                if (
                    "regex" in fn.name.lower()
                    or "replace" in fn.description.lower()
                )
            ]
            if regex_fns:
                filtered = regex_fns

        operator_families: dict[str, frozenset[str]] = {
            "multiply": frozenset({"multiply", "product", "times", "veces"}),
            "add": frozenset({"add", "sum", "plus", "total", "mas", "más"}),
            "subtract": frozenset({
                "subtract", "minus", "less", "difference", "menos",
            }),
            "root": frozenset({"root", "sqrt", "radical"}),
            "divide": frozenset({"divide", "divided", "division"}),
        }
        for keywords in operator_families.values():
            if any(
                re.search(r"\b" + re.escape(kw) + r"\b", low)
                for kw in keywords
            ):
                family_fns = [
                    fn for fn in filtered
                    if any(
                        kw in fn.name.lower() or kw in fn.description.lower()
                        for kw in keywords
                    )
                ]
                if not family_fns:
                    return None
                break

        ask = self.build_selection_prompt(prompt, filtered)
        base = self.build_selection_prompt("...", filtered)
        ask_ids = self.encode(ask)
        base_ids = self.encode(base)

        has_math_signal = bool(math_signals.search(prompt))

        candidates: list[tuple[Any | None, str]] = [
            (fn, fn.name) for fn in filtered
        ]

        pre_available = len(self.extract_numbers(prompt))
        any_fn_satisfies = any(
            0 < sum(
                1 for pd in fn.parameters.values()
                if self.is_numeric(pd.type)
            ) <= pre_available
            for fn in filtered
        )
        if not (has_math_signal and any_fn_satisfies):
            candidates.append((None, "Function not found"))

        scored: list[tuple[Any | None, float]] = []
        for obj, text in candidates:
            tokens = self.encode(text)
            score = (
                self.sequence_logprob(ask_ids, tokens)
                - self.sequence_logprob(base_ids, tokens)
            )
            if obj is not None:
                score += self.keyword_overlap(prompt, obj) * 0.12
            scored.append((obj, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_obj, best_score = scored[0]

        if best_obj is None:
            return None

        required_numeric = sum(
            1
            for pd in best_obj.parameters.values()
            if self.is_numeric(pd.type)
        )
        available_numeric = len(self.extract_numbers(prompt))

        all_string_fn = all(
            self.is_string(pd.type)
            for pd in best_obj.parameters.values()
        )
        if all_string_fn:
            prompt_words = set(LOWER_WORDS_RE.findall(prompt.lower()))
            instruction_only = prompt_words <= (
                INSTRUCTION_VERBS | self.IGNORED_WORDS
                | {"string", "word", "text", "the", "a"}
            )
            if instruction_only and not QUOTED_RE.search(prompt):
                return None

        if len(scored) > 1:
            gap = best_score - scored[1][1]
            has_explicit_arg = bool(QUOTED_RE.search(prompt))
            if (
                has_math_signal
                and required_numeric > 0
                and available_numeric >= required_numeric
            ):
                pass
            elif has_explicit_arg and gap >= 0.05:
                pass
            elif gap < 0.18:
                return None

        if required_numeric > 0 and available_numeric < required_numeric:
            return None

        return best_obj

    # Candidate Extraction

    def extract_numbers(self, prompt: str) -> list[int | float]:
        """Extract numeric candidates in order of appearance.

        Handles digits, signed digits, "minus/negative <digit>",
        "minus/negative <word>", and English number words up to twelve.
        All values are returned sorted by their position in the prompt.
        """
        low = prompt.lower()
        matches: list[tuple[int, int | float]] = []

        negative_prefix = re.compile(
            r"\b(minus|negative)\s+(\d+\.?\d*)\b",
            re.IGNORECASE,
        )
        covered: set[int] = set()
        for m in negative_prefix.finditer(prompt):
            val_str = m.group(2)
            val: int | float = (float(val_str)
                                if "." in val_str else int(val_str))
            matches.append((m.start(), -val))
            covered.update(range(m.start(2), m.start(2) + len(m.group(2))))

        for m in re.finditer(r"(?<![a-zA-Z])-?\d+\.?\d*", prompt):
            if any(p in covered for p in range(m.start(), m.end())):
                continue
            raw = m.group()
            matches.append(
                (m.start(), float(raw) if "." in raw else int(raw))
            )

        negative_word = re.compile(
            r"\b(minus|negative)\s+("
            + "|".join(re.escape(w) for w in self.WORD_TO_NUM)
            + r")\b",
            re.IGNORECASE,
        )
        negative_word_hits: set[str] = set()
        for m in negative_word.finditer(low):
            word = m.group(2).lower()
            matches.append((m.start(), -self.WORD_TO_NUM[word]))
            negative_word_hits.add(word)

        for word, value in self.WORD_TO_NUM.items():
            if word in negative_word_hits:
                continue
            wm = re.search(r"\b" + re.escape(word) + r"\b", low)
            if wm:
                matches.append((wm.start(), value))

        matches.sort(key=lambda x: x[0])
        return [v for _, v in matches]

    def extract_strings(self, prompt: str) -> list[str]:
        """Extract string candidates from a prompt.

        Returns candidates in priority order:
        1. Quoted strings — highest confidence, user-delimited.
        2. Tail of an "in <text>" clause at the end of the prompt.
        3. N-grams and single words — fallback only, skipped when
           higher-priority candidates exist.
        """
        found: list[str] = []

        for a, b in QUOTED_RE.findall(prompt):
            text = a if a else b
            if text.strip():
                found.append(text.strip())

        in_tail = re.search(
            r"\bin\s+([A-Za-zÀ-ÿ0-9][^\n]{1,80})$",
            prompt.strip(),
            re.IGNORECASE,
        )
        if in_tail and not found:
            tail = in_tail.group(1).strip()
            if tail and not self.is_invalid_string(tail):
                found.append(tail)

        if not found:
            words = LOWER_WORDS_RE.findall(prompt)
            for size in (4, 3, 2):
                for i in range(len(words) - size + 1):
                    phrase = " ".join(words[i:i + size])
                    if phrase.lower() in self.IGNORED_WORDS:
                        continue
                    if self.is_invalid_string(phrase):
                        continue
                    if words[i].lower() in INSTRUCTION_VERBS:
                        continue
                    found.append(phrase)
            for word in words:
                if word.lower() not in self.IGNORED_WORDS:
                    if not self.is_invalid_string(word):
                        found.append(word)

        found.append("")

        seen: set[str] = set()
        result: list[str] = []
        for item in found:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def extract_regex_candidates(self, prompt: str) -> list[str]:
        """Return concept-based regex patterns mentioned in the prompt.

        Scans for CONCEPT_MAP keys and returns the corresponding regex
        values. Used to populate the candidate pool for regex parameters.
        """
        low = prompt.lower()
        found: list[str] = []
        for key, value in self.CONCEPT_MAP.items():
            if re.search(r"\b" + re.escape(key.lower()) + r"\b", low):
                if value not in found:
                    found.append(value)
        return found

    def extract_replacement_candidates(self, prompt: str) -> list[str]:
        """Return replacement candidates extracted from the prompt.

        Priority:
        1. Token or quoted string immediately after 'with' or 'por'.
           ALL-CAPS tokens are preserved as-is; others are resolved
           through CONCEPT_MAP.
        2. Any ALL-CAPS word in the prompt (e.g. NUMBERS, NUM).
        """
        found: list[str] = []

        with_match = re.search(
            r"\b(?:with|por)\s+"
            r"(?:'([^']*)'|\"([^\"]*)\"|([A-Za-z0-9_*|/\-]+))",
            prompt,
            re.IGNORECASE,
        )
        if with_match:
            token = (
                with_match.group(1)
                or with_match.group(2)
                or with_match.group(3)
                or ""
            ).strip()
            if token:
                normalized = (
                    token if token.isupper()
                    else self.CONCEPT_MAP.get(token.lower(), token)
                )
                if normalized not in found:
                    found.append(normalized)

        for word in re.findall(r"\b[A-Z]{2,}\b", prompt):
            if word not in found:
                found.append(word)

        return found

    def build_candidates(self, prompt: str, param_type: str,
                         param_name: str = "",
                         chosen: dict[str, Any] | None = None,
                         ) -> list[Any]:
        """Build the candidate list for a single parameter.

        Numeric parameters return values extracted from the prompt.
        String parameters are handled by role:

        - regex: concept patterns from CONCEPT_MAP are prepended as
          high-priority candidates.
        - replacement: the 'with'-token and ALL-CAPS labels are
          prepended, and regex patterns are excluded from the base pool.
        - other: the generic string pool with source_string excluded.

        source_string is always excluded from regex and replacement
        candidates to avoid unnecessary model inferences.
        """
        if self.is_numeric(param_type):
            values = self.extract_numbers(prompt)
            return values if values else [0]

        if self.is_string(param_type):
            excluded: set[str] = set()
            if chosen and param_name in {"regex", "replacement"}:
                src = chosen.get("source_string")
                if src is not None:
                    excluded.add(str(src))
                    excluded.add(str(src).lower())

            if param_name == "regex":
                priority = self.extract_regex_candidates(prompt)
                base = [
                    v for v in self.extract_strings(prompt)
                    if str(v) not in excluded and v not in priority
                ]
                return priority + base

            if param_name == "replacement":
                priority = self.extract_replacement_candidates(prompt)
                if not priority and not QUOTED_RE.search(prompt):
                    return [""]
                regex_patterns = set(self.extract_regex_candidates(prompt))
                base = [
                    v for v in self.extract_strings(prompt)
                    if (
                        str(v) not in excluded
                        and v not in priority
                        and v not in regex_patterns
                    )
                ]
                return priority + base if (priority + base) else [""]

            str_values = self.extract_strings(prompt)
            if excluded:
                str_values = [v for v in str_values if str(v) not in excluded]
            return str_values if str_values else [""]

        return [""]

    # Parameter Ranking

    def build_param_prompt(self, prompt: str, fn: Any,
                           param_name: str,
                           chosen: dict[str, Any]) -> str:
        """Build the context prompt used for candidate ranking.

        The prompt ends just before the model would write the chosen
        value, so the logit vector directly scores each candidate.
        """
        params_text = ", ".join(
            f"{k}:{v.type}" for k, v in fn.parameters.items()
        )
        chosen_text = "\n".join(
            f"{k}={v}" for k, v in chosen.items()
        ) or "none"
        return (
            "Choose EXACTLY ONE best candidate.\n\n"
            "ROLES:\n"
            "source_string=INPUT text to modify (full quoted string).\n"
            "regex=PATTERN to find (concept or literal). BEFORE 'with'.\n"
            "replacement=OUTPUT substitution. AFTER 'with'."
            " Never a preposition.\n\n"
            "RULE: 'replace X with Y' -> regex=X, replacement=Y\n\n"
            "EXAMPLES:\n"
            "replace vowels with *\n"
            "  source_string='' regex=[aeiouAEIOU] replacement=*\n\n"
            "Replace all numbers in \"Hello 34\" with NUMBERS\n"
            "  source_string='Hello 34' regex=[0-9] replacement=NUMBERS\n"
            "  NEVER replacement=[0-9]\n\n"
            "Replace consonants in 'Programming is fun' with *\n"
            "  source_string='Programming is fun'"
            " regex=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z] replacement=*\n"
            "  NEVER source_string='Programming'\n\n"
            "Substitute 'rat' with 'dog' in 'The rat sat on the mat'\n"
            "  source_string='The rat sat on the mat' regex=rat"
            " replacement=dog\n\n"
            "replace everything like a vowel ->"
            " source_string='' regex=[aeiouAEIOU] replacement=''\n\n"
            "CONCEPTS: vowels=[aeiouAEIOU] numbers=[0-9]"
            " consonants=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z] asterisks=*\n"
            "ALL-CAPS label (NUMBERS, NUM): keep exact.\n"
            "No source text given -> source_string=''\n"
            "Math: first number->first param. Negatives valid.\n\n"
            f"Function: {fn.name}({params_text})\n"
            f"Request: {prompt}\n"
            f"Chosen so far:\n{chosen_text}\n"
            f"Need: {param_name}\n"
            "ANSWER:\n"
        )

    def rank_candidate(self, prompt: str, fn: Any,
                       param_name: str, candidates: list[Any],
                       chosen: dict[str, Any],
                       param_index: int = -1) -> Any:
        """Rank parameter candidates and return the best one.

        Scores each candidate by its mean log-probability relative to a
        neutral baseline prompt, then adjusts the score with heuristic
        bonuses and penalties specific to each parameter role.

        param_index is the ordinal position of this parameter among the
        numeric parameters of fn, used to apply a positional boost.
        Pass -1 for string parameters.
        """
        ask = self.build_param_prompt(prompt, fn, param_name, chosen)
        base = self.build_param_prompt("...", fn, param_name, chosen)
        ask_ids = self.encode(ask)
        base_ids = self.encode(base)

        best = candidates[0]
        best_score = float("-inf")

        quoted_in_prompt: set[str] = set()
        for a, b in QUOTED_RE.findall(prompt):
            t = (a if a else b).strip()
            if t:
                quoted_in_prompt.add(t)
                quoted_in_prompt.add(t.lower())

        concept_values: set[str] = {
            v.lower() for v in self.CONCEPT_MAP.values()
        }

        for candidate in candidates:
            text = str(candidate)
            score = (
                self.sequence_logprob(ask_ids, self.encode(text))
                - self.sequence_logprob(base_ids, self.encode(text))
            )
            low = text.lower()

            if candidate in chosen.values():
                score -= 0.7

            if param_index >= 0 and isinstance(candidate, (int, float)):
                try:
                    pos = candidates.index(candidate)
                    if pos == param_index:
                        score += 0.50 if candidate < 0 else 0.30
                    else:
                        score -= 0.15
                except ValueError:
                    pass

            if param_name == "source_string":
                if text in quoted_in_prompt:
                    score += 1.20 + len(text) * 0.01
                if " " in text:
                    score += 0.35
                if len(text) > 8:
                    score += 0.25
                if text == "":
                    score += 0.15
                if text != "" and not quoted_in_prompt:
                    score -= 2.50

            if param_name == "regex":
                if low in self.CONCEPT_MAP:
                    score += 0.60
                if len(text) < 12:
                    score += 0.10

            if param_name == "replacement":
                first_word = low.split()[0] if low.split() else ""
                if first_word in REPLACEMENT_STOPWORDS:
                    score -= 1.50
                if low in self.CONCEPT_MAP:
                    score += 0.45
                if low.isupper():
                    score += 0.45
                if low in concept_values:
                    score -= 1.80
                if candidates and text == str(candidates[0]):
                    score += 0.50

            if score > best_score:
                best_score = score
                best = candidate

        return best

    # Parameter Extraction

    def cast_value(self, value: Any, param_type: str,
                   param_name: str = "") -> Any:
        """Cast a raw candidate value to its declared parameter type.

        Numeric types are coerced to int or float. String types are
        normalised through CONCEPT_MAP for regex and replacement roles.
        ALL-CAPS replacement values are returned as-is.
        """
        if self.is_numeric(param_type):
            if isinstance(value, (int, float)):
                return value
            try:
                return float(value) if "." in str(value) else int(value)
            except ValueError:
                return 0

        if self.is_string(param_type):
            text = str(value)
            if param_name in {"regex", "replacement"}:
                if param_name == "replacement" and text.isupper():
                    return text
                return self.normalize_text(text)
            return text

        return value

    def extract_parameters(self, prompt: str, fn: Any) -> dict[str, Any]:
        """Extract all parameter values for a matched function.

        Numeric parameters are assigned deterministically by order of
        appearance in the prompt. String parameters with well-known
        roles (source_string, replacement, regex) are assigned through
        targeted deterministic extractors before the ranking loop runs.
        Any remaining parameters fall through to logit-based ranking.
        """
        params: dict[str, Any] = {}
        numeric_param_index: dict[str, int] = {}
        idx = 0
        for pname, pdef in fn.parameters.items():
            if self.is_numeric(pdef.type):
                numeric_param_index[pname] = idx
                idx += 1

        numeric_ordered = [
            pname for pname, pdef in fn.parameters.items()
            if self.is_numeric(pdef.type)
        ]
        if numeric_ordered:
            numbers = self.extract_numbers(prompt)
            if len(numbers) >= len(numeric_ordered):
                for i, pname in enumerate(numeric_ordered):
                    params[pname] = numbers[i]

        string_params: set[str] = {
            pname for pname, pdef in fn.parameters.items()
            if self.is_string(pdef.type)
        }

        quoted: list[str] = [
            (a if a else b).strip()
            for a, b in QUOTED_RE.findall(prompt)
            if (a if a else b).strip()
        ]

        if "source_string" in string_params:
            has_in_empty = bool(re.search(
                r"\bin\s+(''\s*$|''\s*[,)]|\"\"$|\"\")",
                prompt,
                re.IGNORECASE,
            ))
            has_in_source = bool(re.search(
                r"\bin\s+['\"]",
                prompt,
                re.IGNORECASE,
            ))
            if has_in_empty:
                params["source_string"] = ""
            elif quoted and has_in_source:
                params["source_string"] = max(quoted, key=len)
            else:
                params["source_string"] = ""

        if "replacement" in string_params:
            replacement_priority = self.extract_replacement_candidates(prompt)
            if replacement_priority:
                with_token = replacement_priority[0]
                params["replacement"] = (
                    with_token if with_token.isupper()
                    else self.normalize_text(with_token)
                )
            elif not quoted:
                params["replacement"] = ""

        if "regex" in string_params:
            regex_priority = self.extract_regex_candidates(prompt)
            if regex_priority:
                params["regex"] = regex_priority[0]

        for param_name, param_def in fn.parameters.items():
            if param_name in params:
                continue
            candidates = self.build_candidates(
                prompt, param_def.type, param_name, params,
            )
            best = self.rank_candidate(
                prompt, fn, param_name, candidates, params,
                numeric_param_index.get(param_name, -1),
            )
            params[param_name] = self.cast_value(
                best, param_def.type, param_name,
            )

        return {
            pname: params[pname]
            for pname in fn.parameters
            if pname in params
        }

    # Main Entrypoint

    def generate_function_call(self, prompt: str,
                               functions: list[Any]) -> dict[str, Any]:
        """Map a natural language prompt to a structured function call.

        Returns a dict with keys 'prompt', 'name', and 'parameters'.
        When no function matches, 'name' is 'Function not found' and
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

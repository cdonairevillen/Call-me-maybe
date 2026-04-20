*This project has been created as part of the 42 curriculum by cdonaire*

# call me maybe

## Introduction to Function Calling in LLMs

---

# Table of Contents

1. Description
2. Instructions
3. Resources

---

# Description

## Overview

This project implements a **function calling system powered by a small language model**.

Instead of answering natural language requests directly, the program transforms user prompts into structured machine-readable function calls.

Example:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40,
    "b": 2
  }
}
```

The objective is not to solve the operation itself, but to identify:

* Which function should be called
* Which typed arguments should be passed

---

## Project Goals

This project was designed to demonstrate:

* Practical use of a small LLM
* Structured output generation
* Reliability over hallucination
* Deterministic engineering around probabilistic systems
* Strong Python architecture
* Static typing, linting, validation, maintainability

---

## Subject Philosophy

The subject requires:

* Python 3.10+
* flake8 compliance
* mypy compliance
* Pydantic models
* JSON input/output
* Use of provided `llm_sdk`
* Function selection using the LLM
* Robust error handling
* Near-perfect accuracy
* 100% valid JSON

The main challenge is that small LLMs are unreliable when generating structured outputs directly.

My solution uses a **hybrid architecture**:

* Use the LLM for semantic routing
* Use deterministic Python for reliability, schema control and JSON generation

---

## Global Workflow

```text
CLI Arguments (argparse)
        ↓
Load JSON files
        ↓
Validate with Pydantic
        ↓
Run Pipeline
        ↓
For each prompt:
    LLM selects function
    Python extracts parameters
    JSON object is built
        ↓
Save final output file
```

---

## Pipeline Architecture

Project responsibilities are separated into modules:

* `main.py` → CLI entry point
* `parser.py` → file loading and validation
* `models.py` → Pydantic schemas
* `pipeline.py` → orchestration
* `llm_engine.py` → function routing + parameter extraction

---

## LLM Engine Design

The engine uses:

```python
Small_LLM_Model()
```

and the vocabulary mapping:

```text
token_id ↔ token_string
```

It also includes:

* encode cache
* logits cache

to improve performance.

---

## Function Selection Algorithm

### Step 1 — Dynamic Prompt Construction

The model receives a routing prompt containing all available functions and the user request.

### Step 2 — Tokenization

The prompt is converted into token IDs using the model tokenizer.

### Step 3 — Candidate Scoring

Instead of generating free text, the system evaluates valid candidates only:

* `fn_add_numbers`
* `fn_reverse_string`
* `fn_greet`
* `Function not found`

### Step 4 — Baseline Comparison

A neutral baseline prompt is scored to reduce token popularity bias.

### Step 5 — Best Candidate Wins

The highest final score is selected.

---

## Constrained Candidate Decoding Approach

The model is not allowed to invent arbitrary outputs.

Its decision space is restricted to:

* known function names
* explicit fallback option

This guarantees:

* no hallucinated function names
* stable routing
* deterministic outputs
* improved reliability

---

## Parameter Extraction Strategy

After the LLM selects the best matching function through constrained candidate scoring, parameters are extracted deterministically.

### Numeric Parameters

Supports:

* integers
* decimals
* negatives
* number words

Examples:

```text
two → 2
zero → 0
-5 → -5
3.14 → 3.14
```

### String Parameters

Supports:

* quoted text
* names
* operands for reverse functions
* generic strings

### Substitution Functions

Supports prompts like:

```text
Replace vowels with *
Replace numbers with NUMBERS
Substitute 'cat' with 'dog'
```

Mapped roles:

* source text
* search pattern
* replacement value

---

## JSON Output Strategy

Instead of asking the model to generate JSON directly, Python builds the final structure:

```python
{
    "prompt": ...,
    "name": ...,
    "parameters": ...
}
```

Then writes it using:

```python
json.dump(...)
```

Benefits:

* always valid JSON
* exact schema control
* no malformed syntax
* no hallucinated keys

---

## Error Handling and Reliability

Handled gracefully:

* missing files
* invalid JSON
* malformed schemas
* unsupported prompts
* empty prompts

Fallback:

```json
{
  "name": "Function not found",
  "parameters": {}
}
```

---

## Performance Analysis

Optimizations:

* encode cache
* logits cache
* deterministic extraction
* modular processing

Targets achieved:

* fast execution
* valid output
* reproducible behavior

---

## Design Decisions

### Why not generate raw JSON with the model?

Because small models are unreliable for strict structured output.

### Why use the LLM only for routing?

Because semantic classification is where LLMs are strongest.

### Why deterministic parsing?

Because typed parameters require reliability.

### Why hybrid architecture?

Because real production systems combine AI + software engineering.

---

## Challenges Faced

* Small model instability
* Ambiguous prompts
* Parameter extraction edge cases
* Balancing AI with deterministic guarantees

---

## Testing Strategy

Tested with:

### Standard Cases

* addition
* multiplication
* greetings
* roots
* reverse strings

### Edge Cases

* empty prompts
* unsupported verbs
* decimals
* negatives
* one-word prompts
* ambiguous requests
* quoted strings

### Validation

All outputs checked for:

* valid JSON
* correct keys
* correct types

---

# Instructions

## Installation

```bash
uv sync
```

---

## Run Program

```bash
uv run python -m src
```

---

## Custom Files

```bash
uv run python -m src \
--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/result.json
```

---

## CLI Arguments Explained

Available arguments:

* `--functions_definition`
* `--input`
* `--output`

If no arguments are provided, default project paths are used.

---

## Debugging

```bash
make debug
```

---

## Linting

```bash
make lint
```

---

# Resources

## Technical References

* Python argparse documentation
* Python json documentation
* Pydantic documentation
* NumPy documentation
* Qwen model documentation
* Structured generation literature
* Function calling architectures in LLM systems

---

## AI Usage Disclosure

AI tools were used as assistants for:

* architecture discussion
* code review
* type hint improvements
* documentation drafting
* adversarial testing ideas

All implementation logic, debugging decisions, architecture choices, and final validation were manually reviewed and understood.

---

## Future Improvements

Possible next versions:

* full token-by-token JSON constrained decoder
* boolean parameters
* nested schemas
* multilingual prompts
* confidence scores
* batch processing
* multi-model support

---

# Final Note

Reliable AI systems are usually not pure AI systems.

They are engineered systems where language models and deterministic software complement each other.

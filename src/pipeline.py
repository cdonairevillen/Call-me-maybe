import json
from typing import List
from .models import FunctionDefinition, FunctionCallOutput, PromptInput
from .llm_engine import LLMEngine


def run_pipeline(prompts: List[PromptInput],
                 functions: List[FunctionDefinition]
                 ) -> List[FunctionCallOutput]:
    """
    Run prompts through the LLM engine.

    Args:
        prompts: List of input prompts.
        functions: Available function definitions.

    Returns:
        List of generated function calls.
    """
    engine = LLMEngine()

    static_texts = [
        "Function not found",
        "...",] + [fn.name for fn in functions]

    for t in static_texts:
        engine.encode(t)

    results: List[FunctionCallOutput] = []

    for prompt in prompts:

        try:
            call = engine.generate_function_call(
                prompt.prompt,
                functions,
            )

            print(
                json.dumps(
                    call,
                    indent=2,
                    ensure_ascii=False,
                )
            )

            result = FunctionCallOutput(
                prompt=prompt.prompt,
                name=call["name"],
                parameters=call.get("parameters", {}),
            )

            results.append(result)

        except (ValueError, KeyError, TypeError) as e:
            print(f"PROMPT ERROR: {prompt.prompt} -> {type(e).__name__}: {e}")
            result = FunctionCallOutput(
                prompt=prompt.prompt,
                name="ERROR",
                parameters={
                    "error_type": type(e).__name__,
                    "message": str(e)
                },
            )
            results.append(result)
            continue

        except Exception as e:
            print(f"UNEXPECTED ERROR {prompt.prompt} -> {repr(e)}")
            result = FunctionCallOutput(
                prompt=prompt.prompt,
                name="ERROR",
                parameters={
                    "error_type": "UnexpectedError",
                    "message": str(e),
                },
            )
            results.append(result)
            continue

    return results

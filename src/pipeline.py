import json
from typing import List
from .models import FunctionDefinition, FunctionCallOutput
from .llm_engine import LLMEngine


def run_pipeline(prompts, functions: List[FunctionDefinition]):

    engine = LLMEngine()

    results = []

    for p in prompts:
        call = engine.generate_function_call(p.prompt, functions)

        print(json.dumps(call, indent=2, ensure_ascii=False))

        result = FunctionCallOutput(
            prompt=p.prompt,
            name=call["name"],
            parameters=call.get("parameters", {}))

        results.append(result)

    return results

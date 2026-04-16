import argparse
import json
import os
from .parser import load_functions, load_prompts
from .pipeline import run_pipeline


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--output",
                        default="data/output/result.json")

    args = parser.parse_args()

    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    results = run_pipeline(prompts, functions)

    output = [r.model_dump() for r in results]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":

    main()

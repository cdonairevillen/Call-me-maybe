import argparse
import json
import os
import sys
from .parser import load_functions, load_prompts
from .pipeline import run_pipeline


def main() -> None:
    """
    Run the function-calling pipeline.

    Loads prompt inputs and function definitions from JSON files,
    executes the pipeline, and writes the results to an output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--output",
                        default="data/output/result.json")

    args = parser.parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)

    except FileNotFoundError as e:
        print(f"Missing file: {e}", file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected setup error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        results = run_pipeline(prompts, functions)

    except Exception as e:
        print(f"PIPELINE ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    output = [r.model_dump() for r in results]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()

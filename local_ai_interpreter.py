import json
import subprocess

OLLAMA_PATH = r"C:\Users\Devarshan\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "qwen2.5:3b"


ALLOWED_OPERATIONS = {
    "remove_duplicates",
    "drop_missing",
    "fill_numeric",
    "fill_categorical",
    "remove_units",
    "standardize_text",
}


def interpret_instruction(instruction):
    prompt = f"""
You are a DataClean instruction interpreter.

Convert the user's instruction into JSON.

Allowed operations:
remove_duplicates
drop_missing
fill_numeric
fill_categorical
remove_units
standardize_text

Return ONLY valid JSON.
Do not explain.
Do not write Python.
Do not add markdown.

Use exactly these JSON structures:

For remove_duplicates:
{{
    "operation": "remove_duplicates",
    "column": null,
    "parameters": {{}}
}}

For drop_missing:
{{
    "operation": "drop_missing",
    "column": null,
    "parameters": {{}}
}}

For fill_numeric:
{{
    "operation": "fill_numeric",
    "column": null,
    "parameters": {{
        "strategy": "mean"
    }}
}}

or:

{{
    "operation": "fill_numeric",
    "column": null,
    "parameters": {{
        "strategy": "median"
    }}
}}

For fill_categorical:
{{
    "operation": "fill_categorical",
    "column": null,
    "parameters": {{
        "strategy": "mode"
    }}
}}

For remove_units:
{{
    "operation": "remove_units",
    "column": "COLUMN_NAME",
    "parameters": {{
        "unit": "UNIT"
    }}
}}

For standardize_text:
{{
    "operation": "standardize_text",
    "column": "COLUMN_NAME",
    "parameters": {{
        "style": "strip"
    }}
}}

Allowed standardize_text styles:
strip
lower
upper
title

Rules:
- Do not invent an operation.
- Do not invent a column.
- Do not invent a unit.
- Do not invent a strategy.
- Do not invent a style.
- If a required parameter is not present, use null.
- Return JSON only.

User instruction:
{instruction}
"""

    result = subprocess.run(
        [
            OLLAMA_PATH,
            "run",
            MODEL_NAME,
        ],
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed: {result.stderr.strip()}")

    output = result.stdout.strip()

    try:
        request = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError(f"Local AI did not return valid JSON: {output}") from error

    operation = request.get("operation")

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Local AI returned an invalid operation: {operation}")

    return request

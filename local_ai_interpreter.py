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


def suggest_operation(issue):
    issue_text = json.dumps(issue)

    prompt = f"""
You are the suggestion engine for DataClean.

Your job is to convert ONE detected data-quality issue
into ONE controlled DataClean operation.

Allowed operations:

remove_duplicates
drop_missing
fill_numeric
fill_categorical
remove_units
standardize_text

Return ONLY valid JSON.

The JSON must have exactly this structure:

{{
    "operation": "...",
    "column": null,
    "parameters": {{}}
}}

Rules:

1. Never invent an operation.
2. Do not modify any data.
3. Do not return explanations.
4. For duplicate_rows, use remove_duplicates.
5. For missing numeric values, use fill_numeric.
6. For fill_numeric, ALWAYS include:
"strategy": "median"
7. For missing categorical/text values, use fill_categorical.
8. For embedded units, use remove_units only when the unit is clearly identifiable.
9. For text inconsistencies, use standardize_text only when the required style is clear.
10. If a safe operation cannot be determined, return:
    {{
        "operation": null,
        "column": null,
        "parameters": {{}},
    }}
Detected issue:

{issue_text}
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
        raise RuntimeError(f"Local AI request failed: {result.stderr.strip()}")

    output = result.stdout.strip()

    try:
        request = json.loads(output)
    except json.JSONDecodeError as error:
        raise ValueError("Local AI did not return valid JSON.") from error

    operation = request.get("operation")

    if operation is not None and operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"Local AI returned unsupported operation: {operation}")

    return request

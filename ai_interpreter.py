import os
import httpx

from google import genai
from google.genai import types
from pydantic import BaseModel


class OperationParameters(BaseModel):
    unit: str | None = None
    strategy: str | None = None
    style: str | None = None


class OperationRequest(BaseModel):
    operation: str
    column: str | None
    parameters: OperationParameters


def interpret_instruction(instruction):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    transport = httpx.HTTPTransport(local_address="0.0.0.0")

    http_client = httpx.Client(
        transport=transport,
        timeout=10.0,
    )

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            httpx_client=http_client,
            timeout=10000,
        ),
    )

    prompt = f"""
You are the instruction interpreter for DataClean.

Your only job is to convert the user's natural-language
data-cleaning instruction into a structured operation.

Allowed operations:

- remove_duplicates
- drop_missing
- fill_numeric
- fill_categorical
- remove_units
- standardize_text

Rules:

- Only use one of the allowed operations.
- Use null for column when the operation does not require a column.
- Use null for parameters that are not required.
- Do not invent operations.
- Do not execute any data-cleaning operation.
- Do not return explanations.

Examples:

User:
Remove duplicate rows

Operation:
remove_duplicates

User:
Remove kg from the weight column

Operation:
remove_units, column=weight, unit=kg

User:
Fill missing numeric values using median

Operation:
fill_numeric, column=null, strategy=median

User:
Standardize the city column to lowercase

Operation:
standardize_text, column=city, style=lower

User instruction:
{instruction}
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OperationRequest,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    if response.parsed is None:
        raise ValueError("AI did not return a valid structured operation.")

    return response.parsed.model_dump()

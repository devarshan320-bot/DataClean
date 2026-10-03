import re


def normalize_instruction(instruction):
    text = instruction.strip()

    if not text:
        raise ValueError("Instruction cannot be empty.")

    text = re.sub(r"\s+", " ", text)

    replacements = [
        (
            r"\bempty\b",
            "missing",
        ),
        (
            r"\bnumbers\b",
            "numeric values",
        ),
        (
            r"\bnumber values\b",
            "numeric values",
        ),
        (
            r"\breplace\b",
            "fill",
        ),
    ]

    for pattern, replacement in replacements:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text

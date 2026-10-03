import re

from instruction_normalizer import normalize_instruction
from operations import create_operation_request


def parse_instruction(instruction):
    text = normalize_instruction(instruction)

    if re.fullmatch(
        r"remove\s+duplicates?",
        text,
        re.IGNORECASE,
    ):
        return create_operation_request("remove_duplicates")

    match = re.match(
        r"remove\s+([a-zA-Z0-9_.]+)\s+from\s+" r"(?:the\s+)?([a-zA-Z0-9_.]+)\s+column",
        text,
        re.IGNORECASE,
    )

    if match:
        unit = match.group(1)
        column = match.group(2)

        return create_operation_request(
            "remove_units",
            column=column,
            parameters={"unit": unit},
        )

    match = re.fullmatch(
        r"fill\s+missing\s+numeric\s+values\s+" r"using\s+(mean|median)",
        text,
        re.IGNORECASE,
    )

    if match:
        strategy = match.group(1).lower()

        return create_operation_request(
            "fill_numeric",
            parameters={"strategy": strategy},
        )

    raise ValueError("Could not understand the instruction.")

OPERATIONS = {
    "remove_duplicates": {
        "description": "Remove duplicate rows from the dataset.",
        "requires_column": False,
        "implemented": True,
    },
    "drop_missing": {
        "description": "Drop rows containing missing values.",
        "requires_column": False,
        "implemented": True,
    },
    "fill_numeric": {
        "requires_column": False,
        "optional_column": True,
        "parameters": {
            "strategy": {
                "required": True,
                "allowed": ["mean", "median"],
            },
        },
        "implemented": True,
    },
    "fill_categorical": {
        "description": "Fill missing values in categorical columns.",
        "requires_column": False,
        "parameters": ["strategy"],
        "allowed_values": {
            "strategy": ["mode"],
        },
        "implemented": True,
    },
    "remove_units": {
        "description": "Remove embedded units from a specified column.",
        "requires_column": True,
        "parameters": ["unit"],
        "implemented": True,
    },
    "standardize_text": {
        "description": "Standardize text values in a specified column.",
        "requires_column": True,
        "parameters": ["style"],
        "allowed_values": {
            "style": ["strip", "lower", "upper", "title"],
        },
        "implemented": True,
    },
    "date_format": {
        "description": "Convert date values to a specified format.",
        "requires_column": True,
        "parameters": ["format"],
        "implemented": False,
    },
    "currency_format": {
        "description": "Format numeric values as a specified currency.",
        "requires_column": True,
        "parameters": ["currency"],
        "implemented": False,
    },
}


def is_supported_operation(operation):
    return operation in OPERATIONS


def is_implemented_operation(operation):
    return operation in OPERATIONS and OPERATIONS[operation]["implemented"]


def get_operation(operation):
    return OPERATIONS.get(operation)


def list_operations():
    return list(OPERATIONS.keys())


def validate_operation(operation, parameters=None):
    if not is_supported_operation(operation):
        return False, f"Unsupported operation: {operation}"

    definition = OPERATIONS[operation]

    if not definition["implemented"]:
        return False, (
            f"Operation '{operation}' is supported but " "not implemented yet."
        )

    parameters = parameters or {}

    for parameter in definition.get("parameters", []):
        if parameter not in parameters:
            return False, (f"Missing required parameter: {parameter}")

    allowed_values = definition.get("allowed_values", {})

    for parameter, allowed in allowed_values.items():
        if parameter in parameters:
            if parameters[parameter] not in allowed:
                return False, (
                    f"Invalid value '{parameters[parameter]}' "
                    f"for parameter '{parameter}'. "
                    f"Allowed values: {', '.join(allowed)}"
                )

    return True, "Operation is valid."


def create_operation_request(
    operation,
    column=None,
    parameters=None,
):
    valid, message = validate_operation_request(
        operation,
        column,
        parameters,
    )

    if not valid:
        raise ValueError(message)

    return {
        "operation": operation,
        "column": column,
        "parameters": parameters or {},
    }


def validate_operation_request(
    operation,
    column=None,
    parameters=None,
):
    if not is_supported_operation(operation):
        return False, f"Unsupported operation: {operation}"

    definition = OPERATIONS[operation]

    if not definition["implemented"]:
        return False, (
            f"Operation '{operation}' is supported but " "not implemented yet."
        )

    if definition["requires_column"] and not column:
        return False, (f"Operation '{operation}' requires a column.")

    if (
        not definition["requires_column"]
        and column
        and not definition.get("optional_column", False)
    ):
        return False, (f"Operation '{operation}' does not use a column.")
    valid, message = validate_operation(
        operation,
        parameters,
    )

    if not valid:
        return False, message

    return True, "Operation request is valid."

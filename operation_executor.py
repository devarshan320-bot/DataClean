from cleaner import (
    clean_dataframe,
    remove_units_from_column,
    standardize_text_column,
    format_date_column,
    standardize_units_in_column,
)

from operations import validate_operation_request


from cleaner import (
    clean_dataframe,
    remove_units_from_column,
    standardize_text_column,
)
from operations import validate_operation_request


def execute_operation(df, request):
    operation = request["operation"]
    column = request.get("column")
    parameters = request.get("parameters", {})

    valid, message = validate_operation_request(
        operation,
        column,
        parameters,
    )

    if not valid:
        raise ValueError(message)

    if column is not None and column not in df.columns:
        raise ValueError(
            f"Column '{column}' was not found in the dataset. "
            f"Available columns: {', '.join(df.columns)}"
        )

    if operation == "remove_duplicates":
        cleaned = df.drop_duplicates().copy()
        changes = []

        duplicate_mask = df.duplicated()

        for index in df.index[duplicate_mask]:
            changes.append(
                {
                    "row": index + 2,
                    "column": "*",
                    "original_value": "Duplicate row",
                    "cleaned_value": "Removed",
                    "reason": "Removed duplicate row",
                }
            )

        return cleaned, changes

    if operation == "remove_units":
        cleaned = df.copy()

        changes = remove_units_from_column(
            cleaned,
            column,
            parameters["unit"],
        )

        return cleaned, changes
        
    if operation == "standardize_units":
        cleaned = df.copy()
        
        changes = standardize_units_in_column(
            cleaned,
            column,
            parameters["target_unit"]
        )
        
        return cleaned, changes

    if operation == "standardize_text":
        cleaned = df.copy()

        changes = standardize_text_column(
            cleaned,
            column,
            parameters["style"],
        )

        return cleaned, changes

    if operation == "date_format":
        cleaned = df.copy()

        changes = format_date_column(
            cleaned,
            column,
            parameters["interpretation"],
            parameters["format"],
        )

        return cleaned, changes

    if operation == "fill_numeric":
        cleaned, stats = clean_dataframe(
            df,
            False,
            False,
            parameters["strategy"],
            None,
            column,
        )

        return cleaned, stats["missing_changes"]

    if operation == "fill_categorical":
        cleaned, stats = clean_dataframe(
            df,
            False,
            False,
            None,
            parameters["strategy"],
            None,
            column,
        )

        return cleaned, stats["missing_changes"]

    raise ValueError(f"Operation '{operation}' has no executor.")

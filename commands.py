import sys

from cleaner import (
    clean_dataframe,
    save_cleaned_csv,
    remove_units_from_column,
    standardize_text_column,
    print_cleaning_log,
)

from operations import validate_operation

from utils import (
    load_csv,
    same_file,
)

from validator import (
    build_rules,
    validate_dataframe,
    print_validation_report,
)


def parse_unit_argument(value):
    if ":" not in value:
        print(
            f"Error: Invalid --remove-units value '{value}'. "
            "Expected COLUMN:UNIT, for example weight:kg."
        )
        sys.exit(1)

    column, unit = value.split(":", 1)

    column = column.strip()
    unit = unit.strip()

    if not column or not unit:
        print(
            f"Error: Invalid --remove-units value '{value}'. "
            "Expected COLUMN:UNIT, for example weight:kg."
        )
        sys.exit(1)

    return column, unit


def parse_standardize_text_argument(value):
    if ":" not in value:
        print(
            f"Error: Invalid --standardize-text value '{value}'. "
            "Expected COLUMN:STYLE, for example gender:title."
        )
        sys.exit(1)

    column, style = value.split(":", 1)

    column = column.strip()
    style = style.strip().lower()

    if not column or not style:
        print(
            f"Error: Invalid --standardize-text value '{value}'. "
            "Expected COLUMN:STYLE, for example gender:title."
        )
        sys.exit(1)

    return column, style


def validate_registered_operation(operation, parameters=None):
    valid, message = validate_operation(
        operation,
        parameters,
    )

    if not valid:
        print(f"Error: {message}")
        sys.exit(1)


def run_clean(
    csv_path,
    remove_duplicates,
    drop_missing,
    fill_numeric,
    fill_categorical,
    remove_units,
    standardize_text,
    output_path,
    preview,
):
    has_fill = fill_numeric is not None or fill_categorical is not None

    has_action = (
        remove_duplicates
        or drop_missing
        or has_fill
        or remove_units
        or standardize_text
    )

    if not has_action:
        print("Error: No cleaning action was specified.")
        print(
            "Use --remove-duplicates, --drop-missing, "
            "--fill-numeric, --fill-categorical, "
            "--remove-units, or --standardize-text."
        )
        sys.exit(1)

    if drop_missing and has_fill:
        print(
            "Error: --drop-missing cannot be used with "
            "--fill-numeric or --fill-categorical."
        )
        print("Choose either dropping missing values or filling them, " "not both.")
        sys.exit(1)

    if remove_duplicates:
        validate_registered_operation("remove_duplicates")

    if drop_missing:
        validate_registered_operation("drop_missing")

    if fill_numeric:
        validate_registered_operation(
            "fill_numeric",
            {"strategy": fill_numeric},
        )

    if fill_categorical:
        validate_registered_operation(
            "fill_categorical",
            {"strategy": fill_categorical},
        )

    if not preview and same_file(csv_path, output_path):
        print(
            "Error: --output must be a new file. "
            "The original CSV will not be overwritten."
        )
        sys.exit(1)

    df = load_csv(csv_path)

    rows_before = len(df)

    cleaned, stats = clean_dataframe(
        df,
        remove_duplicates,
        drop_missing,
        fill_numeric,
        fill_categorical,
    )

    changes = []

    changes.extend(stats["missing_changes"])
    changes.extend(stats["duplicate_changes"])

    if remove_units:
        remove_column, remove_unit = parse_unit_argument(remove_units)

        validate_registered_operation(
            "remove_units",
            {"unit": remove_unit},
        )

        if remove_column not in cleaned.columns:
            print(f"Error: Column '{remove_column}' does not exist.")
            sys.exit(1)

        unit_changes = remove_units_from_column(
            cleaned,
            remove_column,
            remove_unit,
        )

        changes.extend(unit_changes)

    if standardize_text:
        text_column, text_style = parse_standardize_text_argument(standardize_text)

        validate_registered_operation(
            "standardize_text",
            {"style": text_style},
        )

        if text_column not in cleaned.columns:
            print(f"Error: Column '{text_column}' does not exist.")
            sys.exit(1)

        text_changes = standardize_text_column(
            cleaned,
            text_column,
            text_style,
        )

        changes.extend(text_changes)

    if preview:
        print("\nCleaning Preview")
        print("----------------")

        if changes:
            print("\nChanges that would be made:")
            print_cleaning_log(changes)
        else:
            print("\nNo changes detected.")

        print("\nNo file was modified.")
        return

    save_cleaned_csv(cleaned, output_path)

    print(f"Rows before: {rows_before}")

    if drop_missing:
        print(f"Rows dropped for missing values: " f"{stats['missing_rows_dropped']}")

    if fill_numeric:
        print(f"Numeric missing values filled with: " f"{fill_numeric}")

    if fill_categorical:
        print(f"Categorical missing values filled with: " f"{fill_categorical}")

    if remove_duplicates:
        print(f"Duplicate rows removed: " f"{stats['duplicate_rows_removed']}")

    if remove_units:
        print(f"Unit removal applied: " f"{remove_units}")

    if standardize_text:
        print(f"Text standardization applied: " f"{standardize_text}")

    print(f"Remaining missing values: " f"{stats['remaining_missing']}")

    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")

    if changes:
        print_cleaning_log(changes)


def run_instruction(
    csv_path,
    instruction,
    output_path,
    preview,
):
    from instruction_parser import parse_instruction
    from operation_executor import execute_operation

    df = load_csv(csv_path)

    request = parse_instruction(instruction)

    print(f"Instruction: {instruction}")
    print(f"Operation: {request['operation']}")

    cleaned, changes = execute_operation(
        df,
        request,
    )

    if preview:
        print("\nCleaning Preview")
        print("----------------")

        if changes:
            print_cleaning_log(changes)
        else:
            print("No changes would be made.")

        print("\nNo file was modified.")
        return

    if same_file(csv_path, output_path):
        print(
            "Error: --output must be a new file. "
            "The original CSV will not be overwritten."
        )
        sys.exit(1)

    save_cleaned_csv(
        cleaned,
        output_path,
    )

    print(f"\nRows before: {len(df)}")
    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")

    print_cleaning_log(changes)


def run_validate(csv_path, args):
    df = load_csv(csv_path)

    rules = build_rules(
        args,
        df,
    )

    results = validate_dataframe(
        df,
        rules,
    )

    print_validation_report(
        df,
        csv_path,
        results,
    )

    total_violations = sum(len(result["indexes"]) for result in results)

    if total_violations == 0:
        sys.exit(0)

    sys.exit(1)


def run_ai_instruction(csv_path, instruction, output_path, preview):
    from ai_interpreter import interpret_instruction
    from operation_executor import execute_operation
    from operations import validate_operation_request

    df = load_csv(csv_path)

    print(f"Instruction: {instruction}")

    request = interpret_instruction(instruction)

    operation = request["operation"]
    column = request.get("column")
    parameters = request.get("parameters", {})

    print(f"Operation: {operation}")

    valid, message = validate_operation_request(
        operation,
        column,
        parameters,
    )

    if not valid:
        print(f"Error: AI generated an invalid operation: {message}")
        sys.exit(1)

    print("Operation validated successfully.")

    cleaned, changes = execute_operation(
        df,
        request,
    )

    if preview:
        print("\nCleaning Preview")
        print("----------------")

        if changes:
            print_cleaning_log(changes)
        else:
            print("No changes would be made.")

        print("\nNo file was modified.")
        return

    if same_file(csv_path, output_path):
        print(
            "Error: --output must be a new file. "
            "The original CSV will not be overwritten."
        )
        sys.exit(1)

    save_cleaned_csv(
        cleaned,
        output_path,
    )

    print(f"\nRows before: {len(df)}")
    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")

    print_cleaning_log(changes)


def run_local_ai_instruction(
    csv_path,
    instruction,
    output_path,
    preview,
):
    from local_ai_interpreter import interpret_instruction
    from operation_executor import execute_operation
    from operations import validate_operation_request

    df = load_csv(csv_path)

    print(f"Instruction: {instruction}")

    request = interpret_instruction(instruction)

    operation = request["operation"]
    column = request.get("column")
    parameters = request.get("parameters", {})

    print(f"Operation: {operation}")

    valid, message = validate_operation_request(
        operation,
        column,
        parameters,
    )

    if not valid:
        print(f"Error: Local AI generated an invalid operation: {message}")
        sys.exit(1)

    if column is not None and column not in df.columns:
        print(f"Error: Column '{column}' was not found in the dataset.")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)

    print("Operation validated successfully.")

    cleaned, changes = execute_operation(
        df,
        request,
    )

    if preview:
        print("\nCleaning Preview")
        print("----------------")

        if changes:
            print_cleaning_log(changes)
        else:
            print("No changes would be made.")

        print("\nNo file was modified.")
        return

    if same_file(csv_path, output_path):
        print(
            "Error: --output must be a new file. "
            "The original CSV will not be overwritten."
        )
        sys.exit(1)

    save_cleaned_csv(
        cleaned,
        output_path,
    )

    print(f"\nRows before: {len(df)}")
    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")

    print_cleaning_log(changes)


def run_analyze(csv_path):
    from quality_analyzer import analyze_quality

    df = load_csv(csv_path)

    issues = analyze_quality(df)

    print(f"Dataset: {csv_path}")

    print("\nDATA QUALITY ANALYSIS")
    print("=====================")

    if not issues:
        print("\nNo quality issues detected.")
        return

    print(f"\nIssues detected: {len(issues)}")

    for issue in issues:
        issue_type = issue["type"]
        column = issue.get("column")

        if issue_type == "missing_values":
            print(f"\nMissing values in '{column}': " f"{issue['count']}")

        elif issue_type == "duplicate_rows":
            print(f"\nDuplicate rows: " f"{issue['count']}")

        elif issue_type == "text_inconsistency":
            print(f"\nText inconsistency in '{column}':")
            print("  Variants: " + ", ".join(issue["variants"]))

        elif issue_type == "date_format_inconsistency":
            print(f"\nDate-format inconsistency in '{column}':")
            print("  Formats: " + ", ".join(issue["formats"]))

        elif issue_type == "embedded_units":
            print(f"\nEmbedded units in '{column}':")
            print("  Values: " + ", ".join(issue["values"]))

    print("\nNo data was modified.")


def run_suggest(csv_path):
    from quality_analyzer import analyze_quality
    from local_ai_interpreter import suggest_operation

    df = load_csv(csv_path)

    issues = analyze_quality(df)

    print(f"Dataset: {csv_path}")

    print("\nAI QUALITY SUGGESTIONS")
    print("=====================")

    if not issues:
        print("\nNo quality issues detected.")
        return

    for issue in issues:
        print("\nDetected issue:")
        print(issue)

        try:
            suggestion = suggest_operation(issue)

        except (RuntimeError, ValueError) as error:
            print(f"AI suggestion failed: {error}")
            continue

        if suggestion["operation"] is None:
            print("Suggestion: No safe operation identified.")
            continue

        print("\nSuggested operation:")
        print(f"  Operation: {suggestion['operation']}")

        if suggestion.get("column") is not None:
            print(f"  Column: {suggestion['column']}")

        parameters = suggestion.get("parameters", {})

        if parameters:
            print(f"  Parameters: {parameters}")

    print("\nNo data was modified.")


def prepare_suggestion(issue, df):
    from local_ai_interpreter import suggest_operation
    from suggestion_resolver import resolve_suggestion
    from operations import validate_operation_request

    suggestion = suggest_operation(issue)

    if suggestion.get("operation") is None:
        raise ValueError("No safe operation was suggested for this issue.")

    resolved = resolve_suggestion(suggestion, issue, df)
    
    if isinstance(resolved, str):
        # Legacy format, treat as review required
        return {
            "issue": issue,
            "resolution_type": "REVIEW_REQUIRED",
            "message": resolved
        }

    resolution_type = resolved.get("resolution_type", "SAFE_FIX")
    
    if resolution_type == "REVIEW_REQUIRED":
        return {
            "issue": issue,
            "resolution_type": "REVIEW_REQUIRED",
            "message": resolved.get("message", "Review required.")
        }

    operation = resolved.get("operation")
    column = resolved.get("column")
    parameters = resolved.get("parameters", {})

    if resolution_type not in ["STANDARDIZATION_AVAILABLE"]:
        valid, message = validate_operation_request(
            operation,
            column,
            parameters,
        )

        if not valid:
            raise ValueError(f"Invalid AI suggestion: {message}")

    return {
        "issue": issue,
        "operation": operation,
        "column": column,
        "parameters": parameters,
        "resolution_type": resolution_type,
        "message": resolved.get("message", ""),
        "available_targets": resolved.get("available_targets", []),
        "detected_units": resolved.get("detected_units", []),
    }


def preview_suggestion(csv_path, issue):
    from operation_executor import execute_operation

    df = load_csv(csv_path)

    suggestion = prepare_suggestion(issue, df)

    request = {
        "operation": suggestion["operation"],
        "column": suggestion["column"],
        "parameters": suggestion["parameters"],
    }

    cleaned, changes = execute_operation(
        df,
        request,
    )

    print("\nAI Suggestion")
    print("-------------")
    print(f"Operation: {suggestion['operation']}")

    if suggestion["column"] is not None:
        print(f"Column: {suggestion['column']}")

    if suggestion["parameters"]:
        print(f"Parameters: {suggestion['parameters']}")

    print("\nCleaning Preview")
    print("----------------")

    if changes:
        print_cleaning_log(changes)
    else:
        print("No changes would be made.")

    print("\nNo file was modified.")

    return cleaned, changes


def apply_suggestion(csv_path, issue, output_path):
    from operation_executor import execute_operation

    df = load_csv(csv_path)

    suggestion = prepare_suggestion(issue, df)

    request = {
        "operation": suggestion["operation"],
        "column": suggestion["column"],
        "parameters": suggestion["parameters"],
    }

    cleaned, changes = execute_operation(
        df,
        request,
    )

    if same_file(csv_path, output_path):
        print(
            "Error: --output must be a new file. "
            "The original CSV will not be overwritten."
        )
        sys.exit(1)

    save_cleaned_csv(
        cleaned,
        output_path,
    )

    print("\nSuggestion approved.")
    print(f"Operation: {suggestion['operation']}")

    if suggestion["column"] is not None:
        print(f"Column: {suggestion['column']}")

    if suggestion["parameters"]:
        print(f"Parameters: {suggestion['parameters']}")

    print(f"\nRows before: {len(df)}")
    print(f"Rows after: {len(cleaned)}")

    print(f"\nSaved cleaned CSV to: {output_path}")

    print_cleaning_log(changes)

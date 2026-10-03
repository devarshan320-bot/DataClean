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

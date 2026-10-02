import sys

from cleaner import (
    clean_dataframe,
    save_cleaned_csv,
    remove_units_from_column,
    print_cleaning_log,
)

from utils import (
    load_csv,
    same_file,
)

from validator import (
    build_rules,
    validate_dataframe,
    print_validation_report,
)


def run_clean(
    csv_path,
    remove_duplicates,
    drop_missing,
    fill_numeric,
    fill_categorical,
    remove_units,
    output_path,
    preview,
):
    has_fill = fill_numeric is not None or fill_categorical is not None
    has_action = (
        remove_duplicates
        or drop_missing
        or has_fill
        or remove_units
    )

    if not has_action:
        print("Error: No cleaning action was specified.")
        print(
            "Use --remove-duplicates, --drop-missing, "
            "--fill-numeric, --fill-categorical, or --remove-units."
        )
        sys.exit(1)

    if drop_missing and has_fill:
        print(
            "Error: --drop-missing cannot be used with "
            "--fill-numeric or --fill-categorical."
        )
        print(
            "Choose either dropping missing values or filling them, "
            "not both."
        )
        sys.exit(1)

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

    changes = stats["duplicate_changes"]

    if remove_units:
        if remove_units not in cleaned.columns:
            print(f"Error: Column '{remove_units}' does not exist.")
            sys.exit(1)

        unit_changes = remove_units_from_column(
            cleaned,
            remove_units,
            "kg",
        )

        changes.extend(unit_changes)

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
        print(
            f"Rows dropped for missing values: "
            f"{stats['missing_rows_dropped']}"
        )

    if fill_numeric:
        print(
            f"Numeric missing values filled with: "
            f"{fill_numeric}"
        )

    if fill_categorical:
        print(
            f"Categorical missing values filled with: "
            f"{fill_categorical}"
        )

    if remove_duplicates:
        print(
            f"Duplicate rows removed: "
            f"{stats['duplicate_rows_removed']}"
        )

    print(
        f"Remaining missing values: "
        f"{stats['remaining_missing']}"
    )

    print(f"Rows after: {len(cleaned)}")

    print(f"Saved cleaned CSV to: {output_path}")

    if changes:
        print_cleaning_log(changes)


def run_validate(csv_path, args):
    df = load_csv(csv_path)

    rules = build_rules(args, df)
    results = validate_dataframe(df, rules)

    print_validation_report(
        df,
        csv_path,
        results,
    )

    if any(result["indexes"] for result in results):
        sys.exit(1)
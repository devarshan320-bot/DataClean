import os
import sys
import re
import pandas as pd

from utils import load_csv, same_file
from analyzer import print_report

from cleaner import (
    clean_dataframe,
    save_cleaned_csv,
    remove_units_from_column,
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
    has_action = remove_duplicates or drop_missing or has_fill or remove_units

    if not has_action:
        print("Error: No cleaning action was specified.")
        print(
            "Use --remove-duplicates, --drop-missing, "
            "--fill-numeric, --fill-categorical, or --remove-units."
        )
        sys.exit(1)

    if drop_missing and has_fill:
        print("Error: --drop-missing cannot be used with --fill-numeric or --fill-categorical.")
        print("Choose either dropping missing values or filling them, not both.")
        sys.exit(1)

    if same_file(csv_path, output_path):
        print("Error: --output must be a new file. The original CSV will not be overwritten.")
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
    unit_changes = []

    if remove_units:
        if remove_units not in cleaned.columns:
            print(f"Error: Column '{remove_units}' does not exist.")
            sys.exit(1)

        unit_changes = remove_units_from_column(
            cleaned,
            remove_units,
            "kg"
        )    

    if preview:
        print("\nCleaning Preview")
        print("----------------")
        if unit_changes:
            print("\nChanges that would be made:")
            for change in unit_changes:
                print(
                    f"Row {change['row']}: "
                    f"{change['column']} "
                    f"{change['original_value']} -> "
                    f"{change['cleaned_value']} "
                    f"({change['reason']})"
                )
                print(f"\nValues changed: {len(unit_changes)}")
        else:
            print("\nNo changes detected.")

        print("\nNo file was modified.")
        return    
    save_cleaned_csv(cleaned, output_path)

    print(f"Rows before: {rows_before}")
    if drop_missing:
        print(f"Rows dropped for missing values: {stats['missing_rows_dropped']}")
    if fill_numeric:
        print(f"Numeric missing values filled with: {fill_numeric}")
    if fill_categorical:
        print(f"Categorical missing values filled with: {fill_categorical}")
    if remove_duplicates:
        print(f"Duplicate rows removed: {stats['duplicate_rows_removed']}")
    print(f"Remaining missing values: {stats['remaining_missing']}")
    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")
    if unit_changes:
        print("\nCleaning Log")
        print("------------")

        for change in unit_changes:
            print(
                f"Row {change['row']}: "
                f"{change['column']} "
                f"{change['original_value']} -> "
                f"{change['cleaned_value']} "
                f"({change['reason']})"
            )

def run_validate(csv_path, args):
    df = load_csv(csv_path)

    rules = build_rules(args, df)
    results = validate_dataframe(df, rules)

    print_validation_report(df, csv_path, results)

    if any(result["indexes"] for result in results):
        sys.exit(1)

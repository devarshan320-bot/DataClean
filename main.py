import argparse
import os
import sys
import re
import pandas as pd

from utils import load_csv
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

from profiler import (
    profile_dataframe,
    profile_dates,
    profile_units,
)

from commands import (
    run_clean,
    run_validate,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="DataClean — inspect, clean, or validate a CSV file."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a CSV file and print a dataset report.",
    )
    inspect_parser.add_argument(
        "csv_path",
        help="Path to the CSV file to inspect.",
    )

    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a CSV file and save the result to a new file.",
    )
    clean_parser.add_argument(
        "csv_path",
        help="Path to the CSV file to clean.",
    )
    clean_parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate rows from the dataset.",
    )
    clean_parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows that contain any missing values.",
    )
    clean_parser.add_argument(
        "--fill-numeric",
        choices=["mean", "median"],
        help="Fill missing values in numeric columns using mean or median.",
    )
    clean_parser.add_argument(
        "--fill-categorical",
        choices=["mode"],
        help="Fill missing values in non-numeric columns using the mode.",
    )
    clean_parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Path for the new cleaned CSV file. The original file is not changed.",
    )
    clean_parser.add_argument(
    "--remove-units",
    help="Remove a specified unit from a column. Format: COLUMN:UNIT, e.g. weight:kg",
    )
    clean_parser.add_argument(
    "--standardize-text",
    help=(
        "Standardize text in a column. "
        "Format: COLUMN:STYLE, "
        "where STYLE is strip, lower, upper, or title."
    ),
  )
    clean_parser.add_argument(
    "--preview",
    action="store_true",
    help="Preview cleaning changes without creating an output file.",
   )
    profile_parser = subparsers.add_parser(
    "profile",
    help="Profile a CSV file for potential data-quality issues"
    )
    profile_parser.add_argument(
    "input_file",
    help="Path to the CSV file"
  )
    validate_parser = subparsers.add_parser(
        "validate",
        help="Check a CSV file against user-specified rules without changing it.",
    )
    validate_parser.add_argument(
        "csv_path",
        help="Path to the CSV file to validate.",
    )
    validate_parser.add_argument(
        "--min",
        action="append",
        default=[],
        metavar="COLUMN=N",
        help="Flag values below N in COLUMN. Repeatable. Example: --min age=0",
    )
    validate_parser.add_argument(
        "--max",
        action="append",
        default=[],
        metavar="COLUMN=N",
        help="Flag values above N in COLUMN. Repeatable. Example: --max marks=100",
    )
    validate_parser.add_argument(
        "--not-null",
        action="append",
        default=[],
        metavar="COLUMN",
        help="Flag missing values in COLUMN. Repeatable.",
    )
    validate_parser.add_argument(
        "--allowed-values",
        action="append",
        default=[],
        metavar="COLUMN=a,b,c",
        help="Flag non-missing values in COLUMN that are not in the comma-separated list.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "inspect":
        df = load_csv(args.csv_path)
        print_report(df, args.csv_path)
    elif args.command == "clean":
        run_clean(
            args.csv_path,
            args.remove_duplicates,
            args.drop_missing,
            args.fill_numeric,
            args.fill_categorical,
            args.remove_units,
            args.standardize_text,
            args.output,
            args.preview,
        )
        
    elif args.command == "profile":
        try:
            df = pd.read_csv(args.input_file)
        except FileNotFoundError:
            print(f"Error: File not found: {args.input_file}")
            return
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        print(f"\nDATA PROFILE: {os.path.basename(args.input_file)}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        profile_dataframe(df)    
    elif args.command == "validate":
        run_validate(args.csv_path, args)

if __name__ == "__main__":
    main()

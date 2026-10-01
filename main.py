import argparse
import os
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="DataClean — inspect a CSV file and print a basic dataset report."
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

    return parser.parse_args()


def load_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(csv_path):
        print(f"Error: '{csv_path}' is not a file.")
        sys.exit(1)

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Error: '{csv_path}' is empty, so it cannot be read as a CSV.")
        sys.exit(1)
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        print(f"Error: Could not read '{csv_path}' as a CSV file.")
        print(f"Details: {exc}")
        sys.exit(1)


def print_report(df, csv_path):
    filename = os.path.basename(csv_path)
    row_count = len(df)
    column_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    print(f"Dataset: {filename}")
    print(f"Rows: {row_count}")
    print(f"Columns: {column_count}")
    print(f"Duplicate rows: {duplicate_rows}")
    print()

    if column_count == 0:
        print("This CSV has no columns.")
        return

    print("Column names:")
    for name in df.columns:
        print(f"  - {name}")
    print()

    header = f"{'Column':<20} {'Dtype':<12} {'Missing':<10} {'Missing %'}"
    print(header)
    print("-" * len(header))

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        if row_count == 0:
            missing_pct = 0.0
        else:
            missing_pct = (missing_count / row_count) * 100
        dtype = str(df[column].dtype)
        print(f"{column:<20} {dtype:<12} {missing_count:<10} {missing_pct:.1f}%")


def main():
    args = parse_args()

    if args.command == "inspect":
        df = load_csv(args.csv_path)
        print_report(df, args.csv_path)


if __name__ == "__main__":
    main()

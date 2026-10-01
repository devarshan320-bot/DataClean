import argparse
import os
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="DataClean — inspect a CSV file or save a cleaned copy."
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
        "--output",
        required=True,
        metavar="PATH",
        help="Path for the new cleaned CSV file. The original file is not changed.",
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


def same_file(path_a, path_b):
    resolved_a = os.path.normcase(os.path.abspath(path_a))
    resolved_b = os.path.normcase(os.path.abspath(path_b))
    return resolved_a == resolved_b


def clean_dataframe(df):
    cleaned = df.drop_duplicates()
    removed = len(df) - len(cleaned)
    return cleaned, removed


def save_cleaned_csv(df, output_path):
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent and not os.path.isdir(parent):
        print(f"Error: The output folder '{parent}' does not exist.")
        sys.exit(1)

    if os.path.isdir(output_path):
        print(f"Error: '{output_path}' is a folder, not a file.")
        sys.exit(1)

    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        print(f"Error: Could not write the cleaned CSV to '{output_path}'.")
        print(f"Details: {exc}")
        sys.exit(1)


def run_clean(csv_path, remove_duplicates, output_path):
    if not remove_duplicates:
        print("Error: No cleaning action was specified.")
        print("Use --remove-duplicates to remove duplicate rows.")
        sys.exit(1)

    if same_file(csv_path, output_path):
        print("Error: --output must be a new file. The original CSV will not be overwritten.")
        sys.exit(1)

    df = load_csv(csv_path)
    rows_before = len(df)
    cleaned, removed = clean_dataframe(df)
    save_cleaned_csv(cleaned, output_path)

    print(f"Rows before: {rows_before}")
    print(f"Duplicate rows removed: {removed}")
    print(f"Rows after: {len(cleaned)}")
    print(f"Saved cleaned CSV to: {output_path}")


def main():
    args = parse_args()

    if args.command == "inspect":
        df = load_csv(args.csv_path)
        print_report(df, args.csv_path)
    elif args.command == "clean":
        run_clean(args.csv_path, args.remove_duplicates, args.output)


if __name__ == "__main__":
    main()

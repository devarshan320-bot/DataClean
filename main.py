import argparse
import os
import sys

import pandas as pd


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


def fill_numeric_missing(df, strategy):
    cleaned = df.copy()
    for column in cleaned.columns:
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            continue
        if strategy == "mean":
            fill_value = cleaned[column].mean()
        else:
            fill_value = cleaned[column].median()
        if pd.isna(fill_value):
            print(
                f"Warning: Column '{column}' has no numeric values to compute a "
                f"{strategy}. Missing values were left unchanged."
            )
            continue
        cleaned[column] = cleaned[column].fillna(fill_value)
    return cleaned


def fill_categorical_missing(df, strategy):
    cleaned = df.copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            continue
        if strategy == "mode":
            modes = cleaned[column].mode()
            if len(modes) == 0:
                print(
                    f"Warning: Column '{column}' has no mode. "
                    "Missing values were left unchanged."
                )
                continue
            fill_value = modes.iloc[0]
            cleaned[column] = cleaned[column].fillna(fill_value)
    return cleaned


def clean_dataframe(df, remove_duplicates, drop_missing, fill_numeric, fill_categorical):
    cleaned = df.copy()
    missing_rows_dropped = 0
    duplicate_rows_removed = 0

    if drop_missing:
        before_drop = len(cleaned)
        cleaned = cleaned.dropna()
        missing_rows_dropped = before_drop - len(cleaned)
    else:
        if fill_numeric:
            cleaned = fill_numeric_missing(cleaned, fill_numeric)
        if fill_categorical:
            cleaned = fill_categorical_missing(cleaned, fill_categorical)

    if remove_duplicates:
        before_dedupe = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        duplicate_rows_removed = before_dedupe - len(cleaned)

    stats = {
        "missing_rows_dropped": missing_rows_dropped,
        "duplicate_rows_removed": duplicate_rows_removed,
        "remaining_missing": int(cleaned.isna().sum().sum()),
    }
    return cleaned, stats


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


def run_clean(
    csv_path,
    remove_duplicates,
    drop_missing,
    fill_numeric,
    fill_categorical,
    output_path,
):
    has_fill = fill_numeric is not None or fill_categorical is not None
    has_action = remove_duplicates or drop_missing or has_fill

    if not has_action:
        print("Error: No cleaning action was specified.")
        print(
            "Use --remove-duplicates, --drop-missing, "
            "--fill-numeric, or --fill-categorical."
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

def parse_column_value(value):
    if "=" not in value:
        print(f"Error: Invalid rule '{value}'. Expected COLUMN=VALUE.")
        sys.exit(1)

    column, rule_value = value.split("=", 1)

    if not column:
        print(f"Error: Invalid rule '{value}'. Column name is missing.")
        sys.exit(1)

    return column, rule_value


def parse_allowed(value):
    column, allowed_values = parse_column_value(value)

    values = allowed_values.split(",")

    if not values or any(value == "" for value in values):
        print(f"Error: Invalid allowed values rule '{value}'.")
        sys.exit(1)

    return column, values


def build_rules(args, df):
    rules = []

    if not (args.min or args.max or args.not_null or args.allowed_values):
        print("Error: No validation rule was specified.")
        print(
            "Use --min, --max, --not-null, or --allowed-values."
        )
        sys.exit(1)

    for value in args.min:
        column, rule_value = parse_column_value(value)

        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist.")
            sys.exit(1)

        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Error: Column '{column}' is not numeric.")
            sys.exit(1)

        try:
            number = float(rule_value)
        except ValueError:
            print(f"Error: '{rule_value}' is not a number.")
            sys.exit(1)

        rules.append(("min", column, number))

    for value in args.max:
        column, rule_value = parse_column_value(value)

        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist.")
            sys.exit(1)

        if not pd.api.types.is_numeric_dtype(df[column]):
            print(f"Error: Column '{column}' is not numeric.")
            sys.exit(1)

        try:
            number = float(rule_value)
        except ValueError:
            print(f"Error: '{rule_value}' is not a number.")
            sys.exit(1)

        rules.append(("max", column, number))

    for column in args.not_null:
        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist.")
            sys.exit(1)

        rules.append(("not-null", column, None))

    for value in args.allowed_values:
        column, allowed_values = parse_allowed(value)

        if column not in df.columns:
            print(f"Error: Column '{column}' does not exist.")
            sys.exit(1)

        rules.append(("allowed-values", column, allowed_values))

    min_values = {}
    max_values = {}

    for rule_type, column, value in rules:
        if rule_type == "min":
            min_values[column] = value
        elif rule_type == "max":
            max_values[column] = value

    for column in min_values:
        if column in max_values and min_values[column] > max_values[column]:
            print(f"Error: Minimum for '{column}' cannot be greater than maximum.")
            sys.exit(1)

    return rules


def validate_dataframe(df, rules):
    results = []

    for rule_type, column, value in rules:
        if rule_type == "min":
            mask = df[column].notna() & (df[column] < value)

        elif rule_type == "max":
            mask = df[column].notna() & (df[column] > value)

        elif rule_type == "not-null":
            mask = df[column].isna()

        else:
            mask = df[column].notna() & ~df[column].isin(value)

        indexes = df.index[mask].tolist()

        results.append(
            {
                "type": rule_type,
                "column": column,
                "value": value,
                "indexes": indexes,
            }
        )

    return results


def print_validation_report(df, csv_path, results):
    filename = os.path.basename(csv_path)
    total_violations = sum(len(result["indexes"]) for result in results)

    print(f"Validation: {filename}")
    print(f"Rows: {len(df)}")
    print(f"Violations: {total_violations}")
    print()

    for result in results:
        rule_type = result["type"]
        column = result["column"]
        value = result["value"]
        indexes = result["indexes"]

        if rule_type == "min":
            description = f"{column} --min {value:g}"

        elif rule_type == "max":
            description = f"{column} --max {value:g}"

        elif rule_type == "not-null":
            description = f"{column} --not-null"

        else:
            allowed = ",".join(value)
            description = f"{column} --allowed-values {allowed}"

        print(f"{description}: {len(indexes)} row(s)")

        if indexes:
            displayed = indexes[:10]

            for index in displayed:
                print(f"  row index {index}")

            if len(indexes) > 10:
                print(f"  ... and {len(indexes) - 10} more")

    print()

    if total_violations == 0:
        print("Result: PASSED")
    else:
        print("Result: FAILED")


def run_validate(csv_path, args):
    df = load_csv(csv_path)

    rules = build_rules(args, df)
    results = validate_dataframe(df, rules)

    print_validation_report(df, csv_path, results)

    if any(result["indexes"] for result in results):
        sys.exit(1)
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
            args.output,
        )
    elif args.command == "validate":
        run_validate(args.csv_path, args)

if __name__ == "__main__":
    main()

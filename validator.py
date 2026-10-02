import argparse
import os
import sys
import re
import pandas as pd
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

import os
import re
import sys

import pandas as pd


def fill_numeric_missing(df, strategy):
    cleaned = df.copy()
    changes = []

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

        missing_indexes = cleaned.index[cleaned[column].isna()]

        for index in missing_indexes:
            changes.append(
                {
                    "row": index + 2,
                    "column": column,
                    "original_value": "NaN",
                    "cleaned_value": fill_value,
                    "reason": f"Filled missing value using {strategy}",
                }
            )

        cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned, changes


def fill_categorical_missing(df, strategy):
    cleaned = df.copy()
    changes = []

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

            missing_indexes = cleaned.index[cleaned[column].isna()]

            for index in missing_indexes:
                changes.append(
                    {
                        "row": index + 2,
                        "column": column,
                        "original_value": "NaN",
                        "cleaned_value": fill_value,
                        "reason": "Filled missing value using mode",
                    }
                )

            cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned, changes


def clean_dataframe(
    df,
    remove_duplicates,
    drop_missing,
    fill_numeric,
    fill_categorical,
    numeric_column=None,
):
    cleaned = df.copy()

    missing_rows_dropped = 0
    duplicate_rows_removed = 0
    changes = []
    duplicate_changes = []
    missing_changes = []

    if drop_missing:
        before_drop = len(cleaned)
        missing_mask = cleaned.isna().any(axis=1)

        for index in cleaned.index[missing_mask]:
            missing_changes.append(
                {
                    "row": index + 2,
                    "column": "*",
                    "original_value": "Row contains missing value",
                    "cleaned_value": "Removed",
                    "reason": "Dropped row containing missing value",
                }
            )

        cleaned = cleaned.dropna()
        missing_rows_dropped = before_drop - len(cleaned)

    else:
        if fill_numeric:
            if numeric_column is None:
                cleaned, numeric_changes = fill_numeric_missing(
                    cleaned,
                    fill_numeric,
                )
                missing_changes.extend(numeric_changes)
            else:
                if numeric_column not in cleaned.columns:
                    raise ValueError(
                        f"Column '{numeric_column}' was not found in the dataset."
                    )

                if not pd.api.types.is_numeric_dtype(cleaned[numeric_column]):
                    raise ValueError(f"Column '{numeric_column}' is not numeric.")

                column_df = cleaned[[numeric_column]].copy()

                column_df, numeric_changes = fill_numeric_missing(
                    column_df,
                    fill_numeric,
                )

                cleaned[numeric_column] = column_df[numeric_column]

                missing_changes.extend(numeric_changes)
        if fill_categorical:
            cleaned, categorical_changes = fill_categorical_missing(
                cleaned,
                fill_categorical,
            )
            missing_changes.extend(categorical_changes)

    if remove_duplicates:
        duplicate_mask = cleaned.duplicated()

        for index in cleaned.index[duplicate_mask]:
            duplicate_changes.append(
                {
                    "row": index + 2,
                    "column": "*",
                    "original_value": "Duplicate row",
                    "cleaned_value": "Removed",
                    "reason": "Removed duplicate row",
                }
            )

        before_dedupe = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        duplicate_rows_removed = before_dedupe - len(cleaned)

    stats = {
        "missing_rows_dropped": missing_rows_dropped,
        "duplicate_rows_removed": duplicate_rows_removed,
        "remaining_missing": int(cleaned.isna().sum().sum()),
        "duplicate_changes": duplicate_changes,
        "missing_changes": missing_changes,
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


def remove_units_from_column(df, column, unit="kg"):
    pattern = re.compile(
        r"^\s*([-+]?\d+(?:\.\d+)?)\s*" + re.escape(unit) + r"\s*$",
        re.IGNORECASE,
    )

    changes = []

    df[column] = df[column].astype(object)

    for index, value in df[column].items():
        if pd.isna(value):
            continue

        original = str(value).strip()
        match = pattern.match(original)

        if match:
            cleaned = float(match.group(1))

            if cleaned.is_integer():
                cleaned = int(cleaned)

            df.at[index, column] = cleaned

            changes.append(
                {
                    "row": index + 2,
                    "column": column,
                    "original_value": original,
                    "cleaned_value": cleaned,
                    "reason": f"Removed {unit} unit",
                }
            )

    return changes


def standardize_text_column(df, column, style):
    if not (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):
        print(
            f"Error: Column '{column}' is not a text column. "
            "Text standardization can only be applied to text columns."
        )
        sys.exit(1)

    changes = []

    for index, value in df[column].items():
        if pd.isna(value):
            continue

        original = str(value)

        if style == "strip":
            cleaned = original.strip()
            reason = "Standardized text by removing leading/trailing whitespace"

        elif style == "lower":
            cleaned = original.strip().lower()
            reason = "Standardized text using lowercase"

        elif style == "upper":
            cleaned = original.strip().upper()
            reason = "Standardized text using uppercase"

        elif style == "title":
            cleaned = original.strip().title()
            reason = "Standardized text using title case"

        else:
            print(f"Error: Unsupported text standardization style '{style}'.")
            print("Supported styles: strip, lower, upper, title.")
            sys.exit(1)

        if original != cleaned:
            df.at[index, column] = cleaned

            changes.append(
                {
                    "row": index + 2,
                    "column": column,
                    "original_value": original,
                    "cleaned_value": cleaned,
                    "reason": reason,
                }
            )

    return changes


def print_cleaning_log(changes):
    if not changes:
        print("\nCleaning Log")
        print("------------")
        print("No changes were made.")
        return

    print("\nCleaning Log")
    print("------------")

    for change in changes:
        print(
            f"Row {change['row']}: "
            f"{change['column']} "
            f"{change['original_value']} -> "
            f"{change['cleaned_value']} "
            f"({change['reason']})"
        )

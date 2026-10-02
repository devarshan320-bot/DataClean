import os
import re
import pandas as pd
import sys
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
def remove_units_from_column(df, column, unit="kg"):
    pattern = re.compile(
        r"^\s*([-+]?\d+(?:\.\d+)?)\s*" + re.escape(unit) + r"\s*$",
        re.IGNORECASE
    )

    changes = []

    # Work with object dtype so numeric values can be inserted.
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

            changes.append({
                "row": index + 2,
                "column": column,
                "original_value": original,
                "cleaned_value": cleaned,
                "reason": f"Removed {unit} unit"
            })

    return changes

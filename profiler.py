import re

import pandas as pd


def profile_dataframe(df):
    print("\nDATA QUALITY REPORT")
    print("===================")

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    print("\nBasic Data Quality")
    print("------------------")
    print(f"Missing values: {missing_total}")
    print(f"Duplicate rows: {duplicate_total}")

    text_issues = profile_text_issues(df)
    date_issues = profile_date_issues(df)
    unit_issues = profile_unit_issues(df)

    print("\nIssue Summary")
    print("-------------")
    print(f"Text inconsistencies: {len(text_issues)}")
    print(f"Date-format issues: {len(date_issues)}")
    print(f"Columns with embedded units: {len(unit_issues)}")

    print("\nText Inconsistencies")
    print("--------------------")

    if not text_issues:
        print("No text inconsistencies detected.")
    else:
        for column, inconsistencies in text_issues.items():
            print(f"\n{column}:")

            for variants in inconsistencies.values():
                print("  " + ", ".join(sorted(variants)))

    print("\nDate Issues")
    print("-----------")

    if not date_issues:
        print("No date-format inconsistencies detected.")
    else:
        for column, formats in date_issues.items():
            print(f"\n{column}:")
            print("  Multiple date formats detected:")

            for date_format in formats:
                print(f"    {date_format}")

    print("\nEmbedded Units / Text")
    print("---------------------")

    if not unit_issues:
        print("No embedded units detected.")
    else:
        for column, values in unit_issues.items():
            print(f"\n{column}:")

            for value in sorted(set(values)):
                print(f"  {value}")

    print("\nNo data was modified.")


def profile_text_issues(df):
    issues = {}

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna().astype(str)

        normalized = {}

        for value in values:
            key = value.strip().lower()
            normalized.setdefault(key, set()).add(value)

        inconsistencies = {
            key: variants for key, variants in normalized.items() if len(variants) > 1
        }

        if inconsistencies:
            issues[column] = inconsistencies

    return issues


def detect_date_format(value):
    value = str(value).strip()

    patterns = [
        (
            r"^\d{1,2}/\d{1,2}/\d{4}$",
            "DD/MM/YYYY or MM/DD/YYYY",
        ),
        (
            r"^\d{1,2}/\d{1,2}/\d{2}$",
            "DD/MM/YY or MM/DD/YY",
        ),
        (
            r"^\d{1,2}-\d{1,2}-\d{4}$",
            "DD-MM-YYYY or MM-DD-YYYY",
        ),
        (
            r"^\d{4}-\d{1,2}-\d{1,2}$",
            "YYYY-MM-DD",
        ),
        (
            r"^[A-Za-z]+\s+\d{1,2},\s+\d{4}$",
            "Month DD, YYYY",
        ),
    ]

    for pattern, name in patterns:
        if re.match(pattern, value):
            return name

    return "Unknown"


def profile_date_issues(df):
    date_columns = []

    for column in df.columns:
        values = df[column].dropna().astype(str).str.strip()

        if values.empty:
            continue

        parsed = pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )

        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.8:
            date_columns.append(column)

    issues = {}

    for column in date_columns:
        values = df[column].dropna().astype(str).str.strip()

        formats = {}

        for value in values:
            date_format = detect_date_format(value)
            formats.setdefault(date_format, []).append(value)

        if len(formats) > 1:
            issues[column] = list(formats.keys())

    return issues


def profile_unit_issues(df):
    unit_pattern = re.compile(
        r"^\s*[-+]?\d+(?:\.\d+)?\s*" r"(kg|kgs|days?|years?)" r"\s*$",
        re.IGNORECASE,
    )

    issues = {}

    for column in df.columns:
        values = df[column].dropna().astype(str).str.strip()

        matches = []

        for value in values:
            if unit_pattern.match(value):
                matches.append(value)

        if matches:
            issues[column] = matches

    return issues


def profile_dates(df):
    issues = profile_date_issues(df)

    if not issues:
        return

    print("\nDate Issues")
    print("-----------")

    for column, formats in issues.items():
        print(f"\n{column}:")
        print("  Multiple date formats detected:")

        for date_format in formats:
            print(f"    {date_format}")


def profile_units(df):
    issues = profile_unit_issues(df)

    if not issues:
        return

    print("\nEmbedded Units / Text")
    print("---------------------")

    for column, values in issues.items():
        print(f"\n{column}:")

        for value in sorted(set(values)):
            print(f"  {value}")

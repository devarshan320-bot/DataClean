import pandas as pd
import re
def profile_dataframe(df):
    print("\nText Inconsistencies")
    print("--------------------")

    found_issue = False

    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = df[column].dropna().astype(str)

        normalized = {}
        for value in values:
            key = value.strip().lower()
            normalized.setdefault(key, set()).add(value)

        inconsistencies = {
            key: variants
            for key, variants in normalized.items()
            if len(variants) > 1
        }

        if inconsistencies:
            found_issue = True
            print(f"\n{column}:")

            for variants in inconsistencies.values():
                print("  " + ", ".join(sorted(variants)))

    if not found_issue:
        print("No text inconsistencies detected.")
    profile_dates(df)
    profile_units(df)
    print("\nNo data was modified.")
    
def detect_date_format(value):
    value = str(value).strip()

    patterns = [
        (r"^\d{1,2}/\d{1,2}/\d{4}$", "DD/MM/YYYY or MM/DD/YYYY"),
        (r"^\d{1,2}/\d{1,2}/\d{2}$", "DD/MM/YY or MM/DD/YY"),
        (r"^\d{1,2}-\d{1,2}-\d{4}$", "DD-MM-YYYY or MM-DD-YYYY"),
        (r"^\d{4}-\d{1,2}-\d{1,2}$", "YYYY-MM-DD"),
        (r"^[A-Za-z]+\s+\d{1,2},\s+\d{4}$", "Month DD, YYYY"),
    ]

    for pattern, name in patterns:
        if re.match(pattern, value):
            return name

    return "Unknown"


def profile_dates(df):
    date_columns = []

    for column in df.columns:
        values = df[column].dropna().astype(str).str.strip()

        if values.empty:
            continue

        parsed = pd.to_datetime(
            values,
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

        valid_ratio = parsed.notna().mean()

        if valid_ratio >= 0.8:
            date_columns.append(column)

    if not date_columns:
        return

    print("\nDate Issues")
    print("-----------")

    for column in date_columns:
        values = df[column].dropna().astype(str).str.strip()

        formats = {}

        for value in values:
            date_format = detect_date_format(value)

            formats.setdefault(date_format, []).append(value)

        if len(formats) > 1:
            print(f"\n{column}:")
            print("  Multiple date formats detected:")

            for date_format in formats:
                print(f"    {date_format}")

    admission_column = None
    discharge_column = None

    for column in date_columns:
        name = column.lower()

        if "admission" in name:
            admission_column = column

        if "discharge" in name:
            discharge_column = column

    if admission_column and discharge_column:
        admission = pd.to_datetime(
            df[admission_column],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

        discharge = pd.to_datetime(
            df[discharge_column],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

        invalid_dates = discharge < admission

        if invalid_dates.any():
            print("\nDate Relationship Issues:")

            for index in df.index[invalid_dates]:
                patient = (
                    df.loc[index, "patient_id"]
                    if "patient_id" in df.columns
                    else index + 1
                )

                print(
                    f"  {patient}: "
                    f"discharge date is earlier than admission date"
                )
def profile_units(df):
    unit_pattern = re.compile(
        r"^\s*[-+]?\d+(?:\.\d+)?\s*(kg|kgs|days?|years?)\s*$",
        re.IGNORECASE
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

    if not issues:
        return

    print("\nEmbedded Units / Text")
    print("---------------------")

    for column, values in issues.items():
        print(f"\n{column}:")

        for value in sorted(set(values)):
            print(f"  {value}")

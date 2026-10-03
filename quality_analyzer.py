from profiler import (
    profile_text_issues,
    profile_date_issues,
    profile_unit_issues,
)


def analyze_quality(df):
    issues = []

    # Missing values
    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            issues.append(
                {
                    "type": "missing_values",
                    "column": column,
                    "count": missing_count,
                }
            )

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        issues.append(
            {
                "type": "duplicate_rows",
                "count": duplicate_count,
            }
        )

    # Text inconsistencies
    text_issues = profile_text_issues(df)

    for column, inconsistencies in text_issues.items():
        variants = []

        for values in inconsistencies.values():
            variants.extend(sorted(values))

        issues.append(
            {
                "type": "text_inconsistency",
                "column": column,
                "variants": sorted(set(variants)),
            }
        )

    # Date-format issues
    date_issues = profile_date_issues(df)

    for column, formats in date_issues.items():
        issues.append(
            {
                "type": "date_format_inconsistency",
                "column": column,
                "formats": formats,
            }
        )

    # Embedded units
    unit_issues = profile_unit_issues(df)

    for column, values in unit_issues.items():
        issues.append(
            {
                "type": "embedded_units",
                "column": column,
                "values": sorted(set(values)),
            }
        )

    return issues

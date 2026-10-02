import os


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

        print(
            f"{column:<20} "
            f"{dtype:<12} "
            f"{missing_count:<10} "
            f"{missing_pct:.1f}%"
        )
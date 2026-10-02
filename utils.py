import os
import sys
import re
import pandas as pd

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
        
def same_file(path_a, path_b):
    resolved_a = os.path.normcase(os.path.abspath(path_a))
    resolved_b = os.path.normcase(os.path.abspath(path_b))
    return resolved_a == resolved_b

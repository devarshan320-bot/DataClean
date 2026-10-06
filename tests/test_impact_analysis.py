import pandas as pd
import numpy as np
from contracts import ReadyContract, MutateOp, Expression, NormalizeTextOp, DropOp, FilterOp, Condition
from contract_executor import execute_contract

def test_impact_unchanged_values():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    # Filter keeps 2 rows. No values actually *changed* in the existing cells.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "filter", "condition": {"operator": "gt", "column": "A", "value": 1}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 0
    assert impact["Rows removed"] == 1

def test_impact_mutate_existing_values():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    # Mutate A = A + 1. All 3 cells in A change.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "mutate", "column": "A", "expression": {"expr": "`A` + 1"}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 3

def test_impact_mutate_partial_change():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    # Mutate A = A * 1. Values don't change.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "mutate", "column": "A", "expression": {"expr": "`A` * 1"}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 0

def test_impact_normalize_text():
    df = pd.DataFrame({"A": [" foo ", "bar", " baz"], "B": [1, 2, 3]})
    # Strip whitespace. Only " foo " and " baz" change. "bar" is unchanged.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "normalize_text", "columns": ["A"], "style": "strip"}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 2

def test_impact_nan_to_nan():
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
    # Mutate A = A + 0. NaN + 0 = NaN. 
    # Only non-NaNs are processed, but they are unchanged since +0.
    # So 0 values changed.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "mutate", "column": "A", "expression": {"expr": "`A` + 0"}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 0

def test_impact_newly_created_columns():
    df = pd.DataFrame({"A": [1, 2, 3]})
    # Mutate B = A + 1. Creates a new column B.
    # Existing columns (A) do not change.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "mutate", "column": "B", "expression": {"expr": "`A` + 1"}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 0
    assert "B" in impact["Columns added"]

def test_impact_row_column_removal():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    # Drop column C. Filter to keep A > 1.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "drop", "columns": ["C"]},
            {"op": "filter", "condition": {"operator": "gt", "column": "A", "value": 1}}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 0
    assert impact["Rows removed"] == 1
    assert "C" in impact["Columns removed"]

def test_impact_mixed_operations():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [" x ", " y ", "z"]})
    # 1. Filter out row 0 (A=1).
    # 2. Normalize text on B (changes " y " to "y" on row 1, "z" stays "z" on row 2).
    # Expected: 1 value changed.
    contract = ReadyContract.model_validate({
        "operations": [
            {"op": "filter", "condition": {"operator": "gt", "column": "A", "value": 1}},
            {"op": "normalize_text", "columns": ["B"], "style": "strip"}
        ]
    })
    _, impact = execute_contract(df, contract)
    assert impact["Values changed"] == 1

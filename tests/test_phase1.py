import pytest
import pandas as pd
from pydantic import ValidationError
from contracts import ReadyContract, Condition
from contract_validator import validate_contract
from contract_executor import execute_contract
from state_manager import HistoryManager

def get_test_df():
    return pd.DataFrame({
        "Revenue Data": [5000, 8000, None, 10000, 2000],
        "Cost": [2000, 5000, 1000, 8000, 1000],
        "Age": [25, 30, 22, 28, 40],
        "City": ["Mysore  ", "BANGALORE", "mysore", "chennai", "Delhi  \t City"],
        "MixedObject": ["string", 42, None, "another string", 3.14]
    })

def test_condition_validation():
    # 'and' requires conditions
    with pytest.raises(ValidationError):
        Condition(operator="and")
    # 'not' requires exactly 1 condition
    with pytest.raises(ValidationError):
        Condition(operator="not", conditions=[Condition(operator="is_null", column="Age"), Condition(operator="is_null", column="Cost")])
    # is_null cannot have value
    with pytest.raises(ValidationError):
        Condition(operator="is_null", column="Age", value=1)
    # between requires 2 values
    with pytest.raises(ValidationError):
        Condition(operator="between", column="Age", value=[10])

def test_malformed_operations():
    # Missing required fields for specific operation
    with pytest.raises(ValidationError):
        ReadyContract.model_validate({"operations": [{"op": "mutate", "expression": {"expr": "1+1"}}]})
    # sort with mismatched ascending
    with pytest.raises(ValidationError):
        ReadyContract.model_validate({"operations": [{"op": "sort", "by": ["Age", "Cost"], "ascending": [True]}]})

def test_nonexistent_backticked_columns():
    contract = ReadyContract.model_validate({"operations": [{"op": "mutate", "column": "test", "expression": {"expr": "`Ghost Column` + 1"}}]})
    errors = validate_contract(contract, ["Cost"])
    assert "Invalid column reference in expression: 'Ghost Column'" in errors[0]

def test_numeric_only_expression_constants():
    contract = ReadyContract.model_validate({"operations": [{"op": "mutate", "column": "test", "expression": {"expr": "Cost + 'hello'"}}]})
    errors = validate_contract(contract, ["Cost"])
    assert "Invalid constant type 'str'" in errors[0]

def test_all_normalize_text_styles():
    df = get_test_df()
    styles = ["strip", "collapse_whitespace", "lower", "upper", "title"]
    for style in styles:
        contract = ReadyContract.model_validate({"operations": [{"op": "normalize_text", "style": style, "columns": ["City"]}]})
        res, _ = execute_contract(df, contract)
        if style == "strip": assert res["City"].iloc[0] == "Mysore"
        elif style == "collapse_whitespace": assert res["City"].iloc[4] == "Delhi City"
        elif style == "lower": assert res["City"].iloc[1] == "bangalore"
        elif style == "upper": assert res["City"].iloc[2] == "MYSORE"
        elif style == "title": assert res["City"].iloc[3] == "Chennai"

def test_normalize_on_non_string_throws():
    df = get_test_df()
    contract = ReadyContract.model_validate({"operations": [{"op": "normalize_text", "style": "strip", "columns": ["Cost"]}]})
    with pytest.raises(TypeError, match="cannot be applied to non-string column"):
        execute_contract(df, contract)

def test_contains_on_mixed_object():
    df = get_test_df()
    # "MixedObject" column contains strings, ints, floats, None.
    # We want to check that string ops like 'contains' safely process only strings and return False for others without crashing.
    contract = ReadyContract.model_validate({"operations": [{"op": "filter", "condition": {"operator": "contains", "column": "MixedObject", "value": "string"}}]})
    res, _ = execute_contract(df, contract)
    assert len(res) == 2
    assert list(res.index) == [0, 3]

def test_select_and_drop_behavior():
    df = get_test_df()
    select_c = ReadyContract.model_validate({"operations": [{"op": "select", "columns": ["Cost", "Age"]}]})
    res_s, impact_s = execute_contract(df, select_c)
    assert list(res_s.columns) == ["Cost", "Age"]
    assert "City" in impact_s["Columns removed"]

    drop_c = ReadyContract.model_validate({"operations": [{"op": "drop", "columns": ["Cost"]}]})
    res_d, impact_d = execute_contract(df, drop_c)
    assert "Cost" not in res_d.columns
    assert impact_d["Columns removed"] == ["Cost"]

def test_history_manager_validation():
    df = get_test_df()
    manager = HistoryManager(df)
    contract = ReadyContract.model_validate({"operations": [{"op": "drop", "columns": ["Nonexistent"]}]})
    with pytest.raises(ValueError, match="failed semantic validation"):
        manager.apply_contract(contract)

def test_preview_matches_approved_execution():
    df = get_test_df()
    contract = ReadyContract.model_validate({"operations": [
        {"op": "normalize_text", "style": "strip", "columns": ["City"]},
        {"op": "mutate", "column": "Margin", "expression": {"expr": "(`Revenue Data` - Cost) / `Revenue Data`"}}
    ]})
    
    preview_res, preview_impact = execute_contract(df.copy(), contract)
    
    manager = HistoryManager(df)
    manager.apply_contract(contract)
    final_res = manager.get_current_df()
    
    pd.testing.assert_frame_equal(preview_res, final_res)
    assert pd.isna(final_res["Margin"].iloc[2]) 
    assert preview_impact["Values changed"] == 1

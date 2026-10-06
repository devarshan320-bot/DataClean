import pytest
import json
import pandas as pd
from ai_provider import AIProvider
from ai_interpreter import NaturalLanguageInterpreter, InterpreterResult
from context_builder import DatasetContext, ColumnContext

class MockProvider(AIProvider):
    def __init__(self, response_dict: dict | str):
        self.response_dict = response_dict
        
    def generate_structured_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> str:
        if isinstance(self.response_dict, str):
            return self.response_dict
        return json.dumps(self.response_dict)

def get_test_context():
    return DatasetContext(
        row_count=10, column_count=4,
        columns=[
            ColumnContext(name="Name", dtype="object", missing_count=0, missing_percentage=0.0, unique_count=10, semantic_role="name/text"),
            ColumnContext(name="Age", dtype="int64", missing_count=0, missing_percentage=0.0, unique_count=8, min_val=10, max_val=80, semantic_role="age"),
            ColumnContext(name="City", dtype="object", missing_count=0, missing_percentage=0.0, unique_count=3, top_categories=["Bangalore", "Delhi", "Mumbai"], semantic_role="categorical"),
            ColumnContext(name="Revenue", dtype="float64", missing_count=0, missing_percentage=0.0, unique_count=10, min_val=100.0, max_val=2000.0, semantic_role="monetary/numeric amount"),
            ColumnContext(name="OrderDate", dtype="datetime64", missing_count=0, missing_percentage=0.0, unique_count=10, semantic_role="date/datetime"),
            ColumnContext(name="CreatedDate", dtype="datetime64", missing_count=0, missing_percentage=0.0, unique_count=10, semantic_role="date/datetime")
        ],
        sample_rows=[]
    )

def get_test_df():
    return pd.DataFrame({
        "Name": ["A", "B", "C"] * 3 + ["J"],
        "Age": [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "City": ["Bangalore", "Delhi", "Mumbai", "Bangalore", "Delhi", "Mumbai", "Bangalore", "Delhi", "Mumbai", "Bangalore"],
        "Revenue": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "OrderDate": pd.to_datetime(["2023-06-01"] * 10),
        "CreatedDate": pd.to_datetime(["2023-05-01"] * 10)
    })

def test_interpret_remove_below_18():
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "remove",
            "target_entity": "everyone",
            "conditions": [{
                "subject": { "reference_type": "implicit_from_value", "original_text": None },
                "operator": "less_than",
                "value": { "original_text": "18", "semantic_type": "number", "parsed_value": 18 }
            }]
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Remove everyone below 18", get_test_context(), get_test_df())
    
    assert res.status == "ready"
    assert res.contract is not None
    assert res.contract.operations[0].op == "filter"
    assert res.contract.operations[0].condition.operator == "not"
    child = res.contract.operations[0].condition.conditions[0]
    assert child.column == "Age"
    assert child.operator == "lt"
    assert child.value == 18

def test_interpret_explicit_order_date():
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "keep",
            "target_entity": "customers",
            "conditions": [{
                "subject": { "reference_type": "explicit_column", "original_text": "OrderDate" },
                "operator": "equals",
                "value": { "original_text": "June", "semantic_type": "month", "parsed_value": "June" }
            }]
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Keep customers whose OrderDate is in June", get_test_context(), get_test_df())
    
    assert res.status == "ready"
    assert res.contract.operations[0].condition.column == "OrderDate"

def test_interpret_vague_date_ambiguity():
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "keep",
            "target_entity": "customers",
            "conditions": [{
                "subject": { "reference_type": "vague_concept", "original_text": "date" },
                "operator": "equals",
                "value": { "original_text": "June", "semantic_type": "month", "parsed_value": "June" }
            }]
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Keep customers whose date is in June", get_test_context(), get_test_df())
    
    assert res.status == "needs_clarification"
    assert "matches multiple columns: OrderDate, CreatedDate" in res.message

def test_interpret_bangalore_more_than_500():
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "keep",
            "target_entity": "customers",
            "conditions": [
                {
                    "subject": { "reference_type": "implicit_from_value", "original_text": None },
                    "operator": "equals",
                    "value": { "original_text": "Bangalore", "semantic_type": "text", "parsed_value": "Bangalore" }
                },
                {
                    "subject": { "reference_type": "vague_concept", "original_text": "revenue" },
                    "operator": "greater_than",
                    "value": { "original_text": "500", "semantic_type": "number", "parsed_value": 500 }
                }
            ],
            "logic": "and"
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Keep Bangalore customers who spent more than 500", get_test_context(), get_test_df())
    
    assert res.status == "ready"
    cond = res.contract.operations[0].condition
    assert cond.operator == "and"
    assert len(cond.conditions) == 2
    assert cond.conditions[0].column == "City"
    assert cond.conditions[1].column == "Revenue"

def test_interpret_remove_duplicates():
    mock_resp = {
        "steps": [{
            "intent": "deduplicate",
            "target_entity": "duplicates"
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Remove duplicates", get_test_context(), get_test_df())
    assert res.status == "ready"
    assert res.contract.operations[0].op == "drop_duplicates"

def test_interpret_multistep():
    mock_resp = {
        "steps": [
            {
                "intent": "deduplicate",
                "target_entity": "duplicates"
            },
            {
                "intent": "sort",
                "target_entity": "dataset",
                "sort_subject": { "reference_type": "explicit_column", "original_text": "Revenue" },
                "sort_direction": "descending"
            }
        ]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Remove duplicates and sort by revenue desc", get_test_context(), get_test_df())
    
    assert res.status == "ready"
    assert len(res.contract.operations) == 2
    assert res.contract.operations[0].op == "drop_duplicates"
    assert res.contract.operations[1].op == "sort"
    assert res.contract.operations[1].by == ["Revenue"]
    assert res.contract.operations[1].ascending == [False]

def test_interpret_multistep_schema_change_fails_gracefully():
    mock_resp = {
        "steps": [
            {
                "intent": "drop",
                "columns_subject": [{ "reference_type": "explicit_column", "original_text": "City" }]
            },
            {
                "intent": "mutate",
                "target_entity": "City",
                "mutate_subject": { "reference_type": "explicit_column", "original_text": "City" },
                "mutation_operation": "lower"
            }
        ]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Drop City and lowercase City", get_test_context(), get_test_df())
    
    assert res.status == "needs_clarification"
    assert "Could not find any column matching reference 'City'" in res.message

def test_interpret_multistep_state_dependent():
    # Step 1 removes everyone under 40
    # Step 2 asks to drop Delhi (by vague concept)
    # The original df has Delhi. But if we remove everyone under 40, do we still have Delhi?
    # Original DF:
    # A 15 Bangalore, B 20 Delhi, C 25 Mumbai, J 30 Bangalore, A 35 Delhi, B 40 Mumbai, C 45 Bangalore, J 50 Delhi, A 55 Mumbai, B 60 Bangalore
    # Let's filter to keep > 40: 
    # C 45 Bangalore, J 50 Delhi, A 55 Mumbai, B 60 Bangalore
    # All 3 cities are still present.
    # What if we filter Age > 55? Only 60 is left, which is Bangalore. Delhi is completely gone from the dataframe!
    # So if step 1 is "Keep Age > 55" and step 2 is "Remove Delhi", step 2 will NOT find Delhi because df['City'].unique() no longer contains 'Delhi'.
    mock_resp = {
        "steps": [
            {
                "intent": "filter",
                "filter_mode": "keep",
                "target_entity": "everyone",
                "conditions": [{
                    "subject": { "reference_type": "explicit_column", "original_text": "Age" },
                    "operator": "greater_than",
                    "value": { "original_text": "55", "semantic_type": "number", "parsed_value": 55 }
                }]
            },
            {
                "intent": "filter",
                "filter_mode": "remove",
                "target_entity": "everyone",
                "conditions": [{
                    "subject": { "reference_type": "implicit_from_value", "original_text": None },
                    "operator": "equals",
                    "value": { "original_text": "Delhi", "semantic_type": "text", "parsed_value": "Delhi" }
                }]
            }
        ]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    df = get_test_df()
    res = interpreter.interpret("test", get_test_context(), df)
    
    # Step 2 shouldn't find Delhi in the scratchpad state because the previous filter wiped it out
    assert res.status == "needs_clarification"
    assert "Could not find any column matching reference 'Delhi'" in res.message
    
    # Original DF is unchanged
    assert len(df) == 10

def test_interpret_unsupported_aggregate():
    mock_resp = {
        "steps": [{
            "intent": "aggregate",
            "target_entity": "spending",
            "aggregate_subject": { "reference_type": "vague_concept", "original_text": "spending" },
            "aggregate_function": "average"
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Calculate average spending", get_test_context(), get_test_df())
    
    assert res.status == "unsupported"
    assert "Aggregations are currently unsupported" in res.message

def test_interpret_invalid_json():
    interpreter = NaturalLanguageInterpreter(MockProvider("```json\n{bad\n```"))
    res = interpreter.interpret("test", get_test_context(), get_test_df())
    assert res.status == "invalid"
    assert "Invalid JSON" in res.errors[0]

def test_malformed_ir_rejected_by_pydantic():
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "conditions": [{
                # Missing subject!
                "operator": "equals",
                "value": { "original_text": "500", "semantic_type": "number", "parsed_value": 500 }
            }]
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("test", get_test_context(), get_test_df())
    assert res.status == "invalid"
    assert "violated schema" in res.errors[0]

# ============================================================
# Tests for the new evidence-accumulation _resolve_reference()
# ============================================================

def _filter_resp(reference_type: str, original_text, operator: str,
                 val_text: str, semantic_type: str, parsed_value,
                 filter_mode: str = "remove"):
    """Helper: build a single-condition filter mock response."""
    return {
        "steps": [{
            "intent": "filter",
            "filter_mode": filter_mode,
            "conditions": [{
                "subject": {"reference_type": reference_type, "original_text": original_text},
                "operator": operator,
                "value": {"original_text": val_text, "semantic_type": semantic_type, "parsed_value": parsed_value}
            }]
        }]
    }

def test_resolve_vague_age_keyword():
    """'age' appears in ref_text -> resolves to Age column via semantic role token."""
    resp = _filter_resp("vague_concept", "age", "less_than", "18", "number", 18)
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Keep people with age < 18", get_test_context(), get_test_df())
    # Age column: semantic_role="age", and "age" appears in role_parts -> identity
    # 18 is within Age's [10, 80] range -> not violated
    assert res.status == "ready"
    op = res.contract.operations[0]
    assert op.condition.conditions[0].column == "Age"

def test_resolve_verbose_vague_no_identity():
    """'the person is younger than 18 years old' contains no strong evidence -> needs_clarification."""
    resp = _filter_resp("vague_concept", "the person is younger than 18 years old",
                        "less_than", "18", "number", 18)
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Remove people younger than 18", get_test_context(), get_test_df())
    # "person", "younger", "old" -> no name match, no category match, no role token match
    # Value 18 is within Age bounds but cannot CREATE a candidate
    assert res.status == "needs_clarification"

def test_resolve_short_vague_person_no_identity():
    """'person' alone has no semantic identity in DatasetContext -> needs_clarification."""
    resp = _filter_resp("vague_concept", "person", "less_than", "18", "number", 18)
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Remove people younger than 18", get_test_context(), get_test_df())
    assert res.status == "needs_clarification"

def test_resolve_implicit_bangalore():
    """'customers from Bangalore' -> City via categorical dataset-value match in ref_text."""
    resp = _filter_resp("implicit_from_value", "customers from Bangalore",
                        "equals", "Bangalore", "text", "Bangalore", filter_mode="keep")
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Keep Bangalore customers", get_test_context(), get_test_df())
    # "bangalore" is in ref_text, "Bangalore" is in City.top_categories -> identity
    assert res.status == "ready"
    op = res.contract.operations[0]
    assert op.condition.column == "City"

def test_resolve_vague_date_ambiguity_regression():
    """'date' in ref_text matches both OrderDate and CreatedDate -> needs_clarification."""
    resp = _filter_resp("vague_concept", "date", "equals", "June", "month", "June",
                        filter_mode="keep")
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Keep entries from June", get_test_context(), get_test_df())
    # "date" is a substring of both OrderDate and CreatedDate -> both get identity -> ambiguous
    assert res.status == "needs_clarification"
    assert "OrderDate" in res.message
    assert "CreatedDate" in res.message

def test_resolve_ambiguous_numeric_value_no_silent_select():
    """Numeric value 150 fits both Age and Revenue -> must not silently pick one."""
    resp = _filter_resp("vague_concept", "amount", "greater_than", "150", "number", 150)
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Filter by amount > 150", get_test_context(), get_test_df())
    # "amount" is in Revenue's role tokens ("monetary/numeric amount") -> identity for Revenue
    # "amount" is NOT in Age's role tokens ("age") -> no identity for Age
    # So only Revenue gets identity, 150 is inside [100, 2000] -> not violated
    # Result: resolves to Revenue
    assert res.status == "ready"
    op = res.contract.operations[0]
    # filter_mode=remove wraps in not -> conditions[0] is the actual leaf
    assert op.condition.conditions[0].column == "Revenue"

def test_resolve_numeric_no_identity_needs_clarification():
    """'the value' has no semantic identity; numeric value alone cannot create a candidate."""
    resp = _filter_resp("vague_concept", "the value", "greater_than", "150", "number", 150)
    interpreter = NaturalLanguageInterpreter(MockProvider(resp))
    res = interpreter.interpret("Filter value > 150", get_test_context(), get_test_df())
    # "the value" has no name match, no category match, no role token match
    # 150 fits Age [10,80]? No, 150 > 80 -> violates. Fits Revenue [100,2000]? Yes but no identity.
    assert res.status == "needs_clarification"


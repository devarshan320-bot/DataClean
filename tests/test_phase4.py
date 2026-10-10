import pytest
import pandas as pd
import tempfile
import os

from ingestion import DataIngester
from context_builder import ContextBuilder
from state_manager import HistoryManager
from ai_interpreter import NaturalLanguageInterpreter, InterpreterResult
from contracts import ReadyContract
from ai_provider import AIProvider

class MockProvider(AIProvider):
    def __init__(self, response_dict: dict | str):
        self.response_dict = response_dict
        
    def generate_structured_json(self, system_prompt: str, user_prompt: str, json_schema: dict) -> str:
        import json
        if isinstance(self.response_dict, str):
            return self.response_dict
        return json.dumps(self.response_dict)

class MockSessionState:
    def __init__(self):
        self.history_manager = None
        self.pending_contract = None
        self.pending_preview_df = None
        self.pending_impact = None
        self.pending_result = None

def get_test_csv_path(content: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write(content)
        return f.name

def test_successful_upload():
    path = get_test_csv_path("Age,Name\n25,Alice\n17,Bob")
    session = MockSessionState()
    
    ingest_res = DataIngester().ingest_csv(path)
    os.remove(path)
    assert ingest_res.status == "success"
    session.history_manager = HistoryManager(ingest_res.dataframe)
    
    assert session.history_manager is not None
    assert len(session.history_manager.get_current_df()) == 2

def test_failed_upload():
    path = get_test_csv_path("Age,Name\n25,Alice,Extra")
    session = MockSessionState()
    
    ingest_res = DataIngester().ingest_csv(path)
    os.remove(path)
    assert ingest_res.status == "needs_clarification"
    # Session shouldn't be initialized
    assert session.history_manager is None

def test_interpret_and_preview_flow():
    df = pd.DataFrame({"Age": [25, 17], "Name": ["Alice", "Bob"]})
    hm = HistoryManager(df)
    
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "keep",
            "target_entity": "everyone",
            "conditions": [{
                "subject": { "reference_type": "implicit_from_value", "original_text": None },
                "operator": "greater_than_or_equals",
                "value": { "original_text": "18", "semantic_type": "number", "parsed_value": 18 }
            }]
        }]
    }
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    ctx = ContextBuilder().build_context(hm.get_current_df())
    res = interpreter.interpret("Keep 18+", ctx, df)
    
    assert res.status == "ready"
    
    from contract_executor import execute_contract
    preview_df, impact = execute_contract(hm.get_current_df().copy(), res.contract)
    
    assert len(hm.get_current_df()) == 2
    assert len(preview_df) == 1
    
    hm.apply_contract(res.contract)
    assert len(hm.get_current_df()) == 1
    
    pd.testing.assert_frame_equal(hm.get_current_df(), preview_df)

def test_reject_flow():
    df = pd.DataFrame({"Age": [25, 17], "Name": ["Alice", "Bob"]})
    hm = HistoryManager(df)
    
    mock_resp = {
        "steps": [{
            "intent": "filter",
            "filter_mode": "keep",
            "target_entity": "everyone",
            "condition": {
                "node_type": "comparison",
                "subject": { "reference_type": "implicit_from_value", "original_text": None },
                "operator": "greater_than_or_equals",
                "value": { "original_text": "18", "semantic_type": "number", "parsed_value": 18 }
            }
        }]
    }
    res = NaturalLanguageInterpreter(MockProvider(mock_resp)).interpret("test", ContextBuilder().build_context(hm.get_current_df()), hm.get_current_df())
    
    # Simulate reject -> Nothing calls hm.apply_contract
    assert len(hm.get_current_df()) == 2

def test_sequential_and_undo():
    df = pd.DataFrame({"Age": [25, 17, 30], "City": ["A", "A", "B"]})
    hm = HistoryManager(df)
    
    c1 = ReadyContract.model_validate({"operations": [{"op": "filter", "condition": {"operator": "gte", "column": "Age", "value": 18}}]})
    hm.apply_contract(c1)
    
    assert len(hm.get_current_df()) == 2
    
    ctx = ContextBuilder().build_context(hm.get_current_df())
    assert ctx.row_count == 2
    
    c2 = ReadyContract.model_validate({"operations": [{"op": "drop", "columns": ["City"]}]})
    hm.apply_contract(c2)
    
    assert "City" not in hm.get_current_df().columns
    
    hm.undo()
    assert "City" in hm.get_current_df().columns
    assert len(hm.get_current_df()) == 2

def test_clarification_no_execution():
    # If the LLM generates an invalid or unsupported intent, we should get invalid/unsupported
    mock_resp = {
        "steps": [{
            "intent": "aggregate",
            "target_entity": "spending",
            "aggregate_subject": { "reference_type": "vague_concept", "original_text": "spending" },
            "aggregate_function": "average"
        }]
    }
    df = pd.DataFrame({"A": [1]})
    interpreter = NaturalLanguageInterpreter(MockProvider(mock_resp))
    res = interpreter.interpret("Clean", ContextBuilder().build_context(df), df)
    
    assert res.status == "unsupported"
    assert res.contract is None

def test_stale_contract_rejection():
    df = pd.DataFrame({"Age": [25, 17], "City": ["A", "B"]})
    hm = HistoryManager(df)
    
    c1 = ReadyContract.model_validate({"operations": [{"op": "drop", "columns": ["City"]}]})
    
    c2 = ReadyContract.model_validate({"operations": [{"op": "drop", "columns": ["City"]}]})
    hm.apply_contract(c2)
    
    with pytest.raises(ValueError, match="failed semantic validation"):
        hm.apply_contract(c1)

def test_apply_contract_dry_run_type_incompatibility_rejected():
    df = pd.DataFrame({"Age": [25, 17], "Name": ["Alice", "Bob"]})
    hm = HistoryManager(df)
    
    initial_history_len = len(hm.get_history())
    initial_df = hm.get_current_df()
    
    # "Age" exists (passes structural validation), but is numeric so normalize_text fails during execution
    bad_contract = ReadyContract.model_validate({
        "operations": [{"op": "normalize_text", "columns": ["Age"], "style": "lower"}]
    })
    
    with pytest.raises(ValueError, match="dry-run execution failed"):
        hm.apply_contract(bad_contract)
        
    # Rejected contract must not be appended to history
    assert len(hm.get_history()) == initial_history_len
    # Current dataframe and original dataframe must remain unchanged
    pd.testing.assert_frame_equal(hm.get_current_df(), initial_df)
    pd.testing.assert_frame_equal(hm.original_df, initial_df)

def test_apply_contract_success_and_valid_execution():
    df = pd.DataFrame({"Age": [25, 17], "Name": ["Alice", "Bob"]})
    hm = HistoryManager(df)
    
    valid_contract = ReadyContract.model_validate({
        "operations": [{"op": "normalize_text", "columns": ["Name"], "style": "lower"}]
    })
    
    hm.apply_contract(valid_contract)
    assert len(hm.get_history()) == 1
    assert hm.get_current_df()["Name"].tolist() == ["alice", "bob"]

def test_apply_contract_dry_run_rejection_preserves_prior_history():
    df = pd.DataFrame({"Age": [25, 17], "Name": ["Alice", "Bob"]})
    hm = HistoryManager(df)
    
    c1 = ReadyContract.model_validate({
        "operations": [{"op": "filter", "condition": {"operator": "gte", "column": "Age", "value": 18}}]
    })
    hm.apply_contract(c1)
    
    assert len(hm.get_history()) == 1
    df_after_c1 = hm.get_current_df()
    assert len(df_after_c1) == 1
    assert df_after_c1["Age"].tolist() == [25]
    
    # Attempt applying an incompatible contract on the filtered dataframe
    bad_contract = ReadyContract.model_validate({
        "operations": [{"op": "normalize_text", "columns": ["Age"], "style": "upper"}]
    })
    
    with pytest.raises(ValueError, match="dry-run execution failed"):
        hm.apply_contract(bad_contract)
        
    # Existing history and state remain intact
    assert len(hm.get_history()) == 1
    pd.testing.assert_frame_equal(hm.get_current_df(), df_after_c1)
    
    # Subsequent undo still functions correctly on prior history
    hm.undo()
    assert len(hm.get_history()) == 0
    assert len(hm.get_current_df()) == 2


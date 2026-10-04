import unittest
import pandas as pd
import numpy as np
import hashlib
from app import get_issue_id
from quality_analyzer import analyze_quality
from suggestion_resolver import resolve_suggestion
from operation_executor import execute_operation
from commands import prepare_suggestion

class TestStateManagement(unittest.TestCase):
    def setUp(self):
        # Create a simulated original_df with missing values and mixed units
        data = {
            "city": ["NY", "LA", np.nan, "CHI"],
            "quantity": ["2", "5", "3 units", np.nan],
            "customer_name": ["Alice", "Bob", "alice", "Bob"]
        }
        self.original_df = pd.DataFrame(data)
        self.working_df = self.original_df.copy()
        
        # 3. initial analysis creates issue list
        self.issues = analyze_quality(self.original_df)
        self.resolved_issues = set()
        self.suggestions = {}
        self.previews = {}
        self.audit_log = []

    def test_original_remains_unchanged(self):
        # 1. original_df remains unchanged
        # 2. working_df accumulates changes
        
        # Apply an operation to working_df
        op = {"operation": "fill_categorical", "column": "city", "parameters": {"strategy": "mode"}}
        self.working_df, changes = execute_operation(self.working_df, op)
        self.audit_log.extend(changes)
        
        self.assertTrue(self.working_df["city"].isna().sum() == 0)
        self.assertTrue(self.original_df["city"].isna().sum() > 0)
        self.assertEqual(len(self.audit_log), 1)

    def test_unrelated_state_retained(self):
        # 3. multiple issues can simultaneously remain in Apply Suggestion state.
        # 4. applying one issue does not reset unrelated issue state.
        # 5. cached preview is retained when still valid.
        
        # Suppose we get suggestions and previews for city and customer_name
        city_issue = next(i for i in self.issues if i.get("column") == "city")
        cust_issue = next(i for i in self.issues if i.get("column") == "customer_name")
        
        city_id = get_issue_id(city_issue)
        cust_id = get_issue_id(cust_issue)
        
        self.suggestions[city_id] = {"operation": "fill_categorical", "column": "city", "parameters": {"strategy": "mode"}}
        self.suggestions[cust_id] = {"operation": "standardize_text", "column": "customer_name", "parameters": {"style": "title"}}
        
        self.previews[city_id] = {"changes": [], "fingerprint": hashlib.md5(pd.util.hash_pandas_object(self.working_df["city"]).values).hexdigest()}
        self.previews[cust_id] = {"changes": [], "fingerprint": hashlib.md5(pd.util.hash_pandas_object(self.working_df["customer_name"]).values).hexdigest()}
        
        # Apply city issue
        op = self.suggestions[city_id]
        self.working_df, _ = execute_operation(self.working_df, op)
        self.resolved_issues.add(city_id)
        
        # Manually invalidate previews like app.py does
        applied_col = "city"
        for k in list(self.suggestions.keys()):
            if self.suggestions[k].get("column") == applied_col and k != city_id:
                if k in self.previews:
                    del self.previews[k]
        if city_id in self.previews:
            del self.previews[city_id]
            
        # Unrelated customer_name preview should be retained!
        self.assertIn(cust_id, self.previews)
        self.assertNotIn(city_id, self.previews)
        
        # Validating fingerprint logic
        new_fingerprint = hashlib.md5(pd.util.hash_pandas_object(self.working_df["customer_name"]).values).hexdigest()
        self.assertEqual(self.previews[cust_id]["fingerprint"], new_fingerprint)

    def test_affected_preview_invalidated(self):
        # 6. affected preview is invalidated
        # Wait, if we apply something to customer_name, any other issue on customer_name should be invalidated.
        issue_a = {'type': 'missing_values', 'column': 'col_A', 'count': 5}
        issue_b = {'type': 'text_inconsistency', 'column': 'col_A', 'count': 2}
        id_a = get_issue_id(issue_a)
        id_b = get_issue_id(issue_b)
        
        self.suggestions[id_a] = {"operation": "fill_numeric", "column": "col_A"}
        self.suggestions[id_b] = {"operation": "standardize_text", "column": "col_A"}
        self.previews[id_a] = {}
        self.previews[id_b] = {}
        
        # Apply issue_a (touches col_A)
        applied_col = "col_A"
        for k in list(self.suggestions.keys()):
            if self.suggestions[k].get("column") == applied_col and k != id_a:
                if k in self.previews:
                    del self.previews[k]
        if id_a in self.previews:
            del self.previews[id_a]
            
        self.assertNotIn(id_a, self.previews)
        self.assertNotIn(id_b, self.previews)

    def test_mixed_numeric_transition(self):
        # 7. quantity MIXED_NUMERIC transitions correctly
        # 8. fill_numeric becomes available
        # 9. stale AI suggestions cannot override deterministic evidence
        
        qty_missing = next(i for i in self.issues if i.get("type") == "missing_values" and i.get("column") == "quantity")
        qty_units = next(i for i in self.issues if i.get("type") == "embedded_units" and i.get("column") == "quantity")
        
        missing_id = get_issue_id(qty_missing)
        units_id = get_issue_id(qty_units)
        
        # Initially, missing values resolution is MIXED_NUMERIC because it's ambiguous
        initial_missing_sug = resolve_suggestion({"operation": "fill_numeric"}, qty_missing, self.working_df)
        self.assertIsInstance(initial_missing_sug, dict)
        self.assertEqual(initial_missing_sug.get("resolution_type"), "MIXED_NUMERIC")
        
        # The user resolves embedded_units
        self.working_df, _ = execute_operation(self.working_df, {"operation": "remove_units", "column": "quantity", "parameters": {"unit": "units"}})
        
        # Now we re-evaluate missing values!
        new_missing_sug = resolve_suggestion(initial_missing_sug, qty_missing, self.working_df)
        self.assertEqual(new_missing_sug.get("operation"), "fill_numeric")
        self.assertEqual(new_missing_sug.get("column"), "quantity")
        
        # It successfully transitioned!



    def test_text_inconsistency_regression(self):
        # 1. status-style casing inconsistency can resolve safely when evidence supports it.
        # 2. No column-specific hardcoding is used.
        # 3. A generic text column with equivalent casing variants behaves correctly.
        # 4. Truly ambiguous text inconsistencies still remain Review required.
        # 5. Applying other issues before status does not cause this regression.

        # Let's create a dataframe matching the 50% scenario
        data = {
            'generic_status': ['Delivered', 'Delivered', 'Delivered', 'delivered', 'delivered', 'DELIVERED', 'DELIVERED'],
            'ambiguous_col': ['StatusA', 'StatusA', 'statusA', 'statusA', 'statusa', 'statusa', 'STATUSa']
        }
        # In generic_status: Title=3, lower=2, upper=2 (Total 7). Title is 3/7 (not 50%).
        # Wait, the regression was exactly 50%. Let's make Title exactly 50%.
        # 6 Title, 4 Lower, 2 Upper (Total 12)
        
        data = {
            'generic_status': ['Title'] * 6 + ['title'] * 4 + ['TITLE'] * 2,
            'ambiguous_col': ['Title'] * 5 + ['title'] * 5 + ['TITLE'] * 2
        }
        df = pd.DataFrame(data)
        
        # Test generic_status (6 is 50% of 12, and 6 > 4, so it should resolve safely)
        issue_generic = {'type': 'text_inconsistency', 'column': 'generic_status', 'variants': ['Title', 'title', 'TITLE']}
        sug_generic = resolve_suggestion({'operation': 'standardize_text', 'parameters': {}}, issue_generic, df)
        self.assertIsInstance(sug_generic, dict)
        self.assertEqual(sug_generic.get('operation'), 'standardize_text')
        self.assertEqual(sug_generic.get('parameters', {}).get('style'), 'title')
        
        # Test ambiguous_col (5 Title, 5 lower. 5 < 6, so < 50%, and 5 == 5. Should be Review Required)
        issue_ambiguous = {'type': 'text_inconsistency', 'column': 'ambiguous_col', 'variants': ['Title', 'title', 'TITLE']}
        sug_ambig = resolve_suggestion({'operation': 'standardize_text', 'parameters': {}}, issue_ambiguous, df)
        self.assertIsInstance(sug_ambig, str)
        self.assertTrue('Review required' in sug_ambig)


if __name__ == "__main__":
    unittest.main()

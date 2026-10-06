import pytest
import pandas as pd
from ingestion import DataIngester
from context_builder import ContextBuilder

def test_ingest_valid_csv(tmp_path):
    f = tmp_path / "valid.csv"
    f.write_text("A,B,C\n1,2,3\n4,5,6", encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "success"
    assert res.dataframe is not None
    assert len(res.dataframe) == 2

def test_ingest_missing_fields_suspends(tmp_path):
    f = tmp_path / "missing.csv"
    f.write_text("A,B,C\n1,2\n4,5,6", encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "needs_clarification"
    assert res.structural_errors[0].issue_type == "missing_fields"

def test_ingest_extra_fields_suspends(tmp_path):
    f = tmp_path / "extra.csv"
    f.write_text("A,B\n1,2,3\n4,5", encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "needs_clarification"
    assert res.structural_errors[0].issue_type == "extra_fields"

def test_ingest_malformed_quotes(tmp_path):
    f = tmp_path / "quotes.csv"
    f.write_text('A,B\n1,"2\n3,4', encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "needs_clarification"
    assert any(e.issue_type == "malformed_quotes" for e in res.structural_errors)

def test_ingest_encoding_failure(tmp_path):
    f = tmp_path / "encoding.csv"
    f.write_bytes(b"A,B\n\xff\xfe,2")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "fatal_error"
    assert res.structural_errors[0].issue_type == "encoding_error"

def test_context_builder_no_mutation():
    df = pd.DataFrame({"A": [1, 2, None], "B": ["x", "y", "x"]})
    df_copy = df.copy(deep=True)
    ctx = ContextBuilder().build_context(df)
    pd.testing.assert_frame_equal(df, df_copy)

def test_context_semantic_hints():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "cat": ["A", "B", "A", "B"],
        "dt": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    })
    ctx = ContextBuilder().build_context(df)
    assert ctx.columns[0].semantic_hint == "Likely Identifier"
    assert ctx.columns[1].semantic_hint == "Likely Categorical"
    assert ctx.columns[2].semantic_hint == "Likely Date"

def test_context_json_serializable():
    df = pd.DataFrame({"A": [1, None, 3], "B": ["x", "y", "z"]})
    ctx = ContextBuilder().build_context(df)
    json_str = ctx.model_dump_json()
    assert "Likely Categorical" in json_str
    assert "null" in json_str

def test_ingest_quoted_comma(tmp_path):
    f = tmp_path / "comma.csv"
    f.write_text('A,B\n1,"2,3"\n4,5', encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "success"
    assert res.dataframe.iloc[0]["B"] == "2,3"

def test_ingest_empty_file(tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "fatal_error"
    assert res.structural_errors[0].issue_type == "empty_file"

def test_safe_type_inference(tmp_path):
    f = tmp_path / "inference.csv"
    f.write_text("Age,Revenue,ID,Mixed,Date\n25,1000,00123,100,2023-01-01\n17,500,00124,unknown,2023-01-02", encoding="utf-8")
    res = DataIngester().ingest_csv(str(f))
    assert res.status == "success"
    df = res.dataframe
    
    import pandas as pd
    assert pd.api.types.is_numeric_dtype(df["Age"]), "Age should be numeric"
    assert pd.api.types.is_numeric_dtype(df["Revenue"]), "Revenue should be numeric"
    assert pd.api.types.is_object_dtype(df["ID"]) or pd.api.types.is_string_dtype(df["ID"]), "ID with leading zeros should stay string"
    assert pd.api.types.is_object_dtype(df["Mixed"]) or pd.api.types.is_string_dtype(df["Mixed"]), "Mixed column should stay string"
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), "Date column should be inferred as datetime"
def test_semantic_inference():
    df = pd.DataFrame({
        "Patient_Age": [25, 30, 40, 110],
        "Yearly_Revenue": [1000.50, 500.00, 2000.0, 1500.0],
        "Employee_ID": ["001", "002", "003", "004"],
        "Order_Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
        "Status": ["A", "B", "A", "C"],
        "Unknown_Metric": [1.1, 2.2, 3.3, 4.4],
        "Mixed_Data": ["100", "200", "unknown", "300"],
        "Contact_Email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com"],
        "Full_Name": ["Alice", "Bob", "Charlie", "David"],
        "Is_Active": [1, 0, 1, 1],
        "Random_Two_Vals": ["Apple", "Orange", "Apple", "Orange"],
        "Department": ["HR", "IT", "HR", "IT"]
    })
    
    ctx = ContextBuilder().build_context(df)
    roles = {c.name: c.semantic_role for c in ctx.columns}
    confs = {c.name: c.semantic_confidence for c in ctx.columns}
    
    assert roles["Patient_Age"] == "age"
    assert roles["Yearly_Revenue"] == "monetary/numeric amount"
    assert roles["Employee_ID"] == "identifier"
    assert roles["Order_Date"] == "date/datetime"
    
    assert roles["Status"] == "categorical" # "status" keyword
    assert roles["Full_Name"] == "name/text" # "name" keyword
    assert roles["Is_Active"] == "boolean/binary" # "is_" keyword and 0/1
    
    assert roles["Unknown_Metric"] == "numeric"
    assert roles["Mixed_Data"] == "identifier" # All unique strings fallback
    assert roles["Contact_Email"] == "email"
    
    # Apple/Orange has 2 values but no binary or categorical name -> unknown
    assert roles["Random_Two_Vals"] == "unknown"
    
    # Department has 2 values and matches categorical keyword -> categorical
    assert roles["Department"] == "categorical"

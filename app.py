import hashlib
import json
import streamlit as st
import pandas as pd
from quality_analyzer import analyze_quality, get_display_issues
from commands import prepare_suggestion
from operation_executor import execute_operation

def get_issue_id(issue):
    issue_str = json.dumps(issue, sort_keys=True)
    return hashlib.md5(issue_str.encode()).hexdigest()



def main():
    st.set_page_config(page_title="DataClean", layout="centered")

    st.title("DataClean")
    st.subheader("Intelligent Data Quality Tool")

    # Initialize session state for DataFrame and issues
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "working_df" not in st.session_state:
        st.session_state.working_df = None
    if "issues" not in st.session_state:
        st.session_state.issues = None
    if "resolved_issues" not in st.session_state:
        st.session_state.resolved_issues = set()
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None
    if "suggestions" not in st.session_state:
        st.session_state.suggestions = {}
    if "previews" not in st.session_state:
        st.session_state.previews = {}
    if "suggestion_errors" not in st.session_state:
        st.session_state.suggestion_errors = {}
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        if st.session_state.last_uploaded_file != uploaded_file.file_id:
            try:
                st.session_state.original_df = pd.read_csv(uploaded_file)
                st.session_state.working_df = st.session_state.original_df.copy()
                st.session_state.last_uploaded_file = uploaded_file.file_id
                st.session_state.issues = None
                st.session_state.resolved_issues = set()
                st.session_state.suggestions = {}
                st.session_state.previews = {}
                st.session_state.suggestion_errors = {}
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if st.session_state.original_df is not None:
        original_df = st.session_state.original_df
        working_df = st.session_state.working_df

        st.header("Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of rows", working_df.shape[0])
        with col2:
            st.metric("Number of columns", working_df.shape[1])
        
        st.write("**Column names:**")
        st.write(", ".join(working_df.columns))

        csv_data = working_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.divider()

        if st.button("Analyze Data Quality"):
            with st.spinner("Analyzing data quality..."):
                st.session_state.issues = analyze_quality(original_df)

        if st.session_state.issues is not None:
            st.header("Analysis Results")
            issues = st.session_state.issues

            if not issues:
                st.success("No quality issues detected!")
            else:
                display_issues = get_display_issues(issues, original_df)
                
                if not display_issues:
                    st.success("No actionable quality issues detected!")
                else:
                    st.warning(f"Detected {len(display_issues)} potential quality issues.")
                    
                    for i, issue in enumerate(display_issues):
                        issue_id = get_issue_id(issue)
                        issue_type = issue.get("type", "unknown")
                        title = issue_type.replace('_', ' ').title()
                    
                        if "column" in issue:
                            title += f" in '{issue['column']}'"
                            
                        # Check if already resolved
                        if issue_id in st.session_state.resolved_issues:
                            with st.expander(f"{i + 1}. ✅ {title} (Resolved)", expanded=False):
                                st.success("This issue has been successfully resolved.")
                            continue

                        # Lightweight applicability check
                        is_resolved = False
                        col = issue.get("column")
                        if col and col not in working_df.columns:
                            is_resolved = True
                        elif issue_type == "missing_values" and col and working_df[col].isna().sum() == 0:
                            is_resolved = True
                        elif issue_type == "duplicate_rows" and working_df.duplicated().sum() == 0:
                            is_resolved = True
                        elif issue_type == "embedded_units" and col:
                            non_nulls = working_df[col].dropna().astype(str).str.strip()
                            import re
                            unit_pattern = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*([a-zA-Z]+)\s*$")
                            if not any(unit_pattern.match(val) for val in non_nulls):
                                is_resolved = True
                                
                        if is_resolved:
                            st.session_state.resolved_issues.add(issue_id)
                            st.rerun()

                        with st.expander(f"{i + 1}. {title}", expanded=True):
                            if issue_type == "missing_values":
                                st.write(f"**Missing count:** {issue.get('count')}")
                            elif issue_type == "duplicate_rows":
                                st.write(f"**Duplicate count:** {issue.get('count')}")
                            elif issue_type == "text_inconsistency":
                                st.write("**Variants found:**")
                                st.write(", ".join(issue.get("variants", [])))
                            elif issue_type == "date_format_inconsistency":
                                st.write("**Formats found:**")
                                st.write(", ".join(issue.get("formats", [])))
                            elif issue_type == "embedded_units":
                                st.write("**Values with units found:**")
                                st.write(", ".join(issue.get("values", [])))
                            else:
                                st.json(issue)
                        
                            st.divider()
                        
                            # AI Suggestion Workflow
                            if issue_type == "date_format_inconsistency":
                                st.write("#### Format Resolution")
                                interp = st.selectbox("Date interpretation (for ambiguous dates):", ["DD-MM-YYYY", "MM-DD-YYYY"], key=f"interp_{issue_id}")
                                target = st.selectbox("Target output format:", ["DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD", "Month DD, YYYY"], key=f"target_{issue_id}")
                            
                                new_suggestion = {
                                    "operation": "date_format",
                                    "column": issue.get("column"),
                                    "parameters": {
                                        "interpretation": interp,
                                        "format": target
                                    }
                                }
                            
                                if issue_id in st.session_state.suggestions:
                                    old_suggestion = st.session_state.suggestions[issue_id]
                                    if old_suggestion.get("parameters") != new_suggestion["parameters"]:
                                        if issue_id in st.session_state.previews:
                                            del st.session_state.previews[issue_id]
                                        
                                st.session_state.suggestions[issue_id] = new_suggestion

                            if issue_id in st.session_state.suggestions:
                                suggestion = st.session_state.suggestions[issue_id]
                                from suggestion_resolver import resolve_suggestion
                                re_evaluated = resolve_suggestion(suggestion, issue, working_df)
                                
                                if isinstance(re_evaluated, dict):
                                    if re_evaluated.get("resolution_type") == "REVIEW_REQUIRED":
                                        st.warning("Previous suggestion is no longer safely applicable. Please get a new AI suggestion.")
                                        if st.button("Re-evaluate", key=f"re_eval_{issue_id}"):
                                            del st.session_state.suggestions[issue_id]
                                            if issue_id in st.session_state.previews:
                                                del st.session_state.previews[issue_id]
                                            st.rerun()
                                        continue
                                    
                                    if re_evaluated.get("resolution_type") != suggestion.get("resolution_type") or re_evaluated.get("operation") != suggestion.get("operation"):
                                        st.session_state.suggestions[issue_id] = re_evaluated
                                        if issue_id in st.session_state.previews:
                                            del st.session_state.previews[issue_id]
                                        st.rerun()
                                        
                                    if "parameters" in suggestion:
                                        for k, v in suggestion["parameters"].items():
                                            if k not in re_evaluated.get("parameters", {}):
                                                if "parameters" not in re_evaluated:
                                                    re_evaluated["parameters"] = {}
                                                re_evaluated["parameters"][k] = v
                                    st.session_state.suggestions[issue_id] = re_evaluated

                            if issue_id not in st.session_state.suggestions:
                                if st.button("Get AI Suggestion", key=f"suggest_btn_{issue_id}"):
                                    with st.spinner("Getting AI suggestion..."):
                                        try:
                                            if issue_id in st.session_state.suggestion_errors:
                                                del st.session_state.suggestion_errors[issue_id]
                                        
                                            # Enrich issue with datatype info to help AI
                                            issue_for_ai = issue.copy()
                                            if "column" in issue and issue["column"] in working_df.columns:
                                                col_dtype = working_df[issue["column"]].dtype
                                                if pd.api.types.is_numeric_dtype(working_df[issue["column"]]):
                                                    issue_for_ai["data_type"] = f"{col_dtype} (numeric)"
                                                else:
                                                    issue_for_ai["data_type"] = f"{col_dtype} (categorical/text)"
                                        
                                            suggestion = prepare_suggestion(issue_for_ai, working_df)
                                            st.session_state.suggestions[issue_id] = suggestion
                                            st.rerun()
                                        except Exception as e:
                                            st.session_state.suggestion_errors[issue_id] = str(e)
                                            st.rerun()
                            
                                if issue_id in st.session_state.suggestion_errors:
                                    st.error(st.session_state.suggestion_errors[issue_id])
                            else:
                                suggestion = st.session_state.suggestions[issue_id]
                                resolution_type = suggestion.get("resolution_type", "SAFE_FIX")
                            
                                if resolution_type == "REVIEW_REQUIRED":
                                    st.warning(suggestion.get("message", "Review required."))
                                    continue
                                
                                if resolution_type == "STANDARDIZATION_AVAILABLE":
                                    st.write("#### Standardization Available")
                                    st.write(f"Detected units: {', '.join(suggestion.get('detected_units', []))}")
                                
                                    do_std = st.radio("Would you like to standardize them?", ["Yes", "No"], key=f"do_std_{issue_id}", index=None)
                                
                                    if do_std == "No":
                                        st.info("Left unchanged.")
                                        continue
                                    elif do_std == "Yes":
                                        target = st.selectbox("Target unit:", suggestion.get("available_targets", []), key=f"target_{issue_id}")
                                        if suggestion.get("parameters", {}).get("target_unit") != target:
                                            if issue_id in st.session_state.previews:
                                                del st.session_state.previews[issue_id]
                                        suggestion["parameters"] = {"target_unit": target}
                                    else:
                                        continue
                                elif resolution_type == "MIXED_NUMERIC":
                                    st.write("#### Mixed Numeric Representation")
                                    st.write(suggestion.get("message", "Some values contain unit labels."))
                                
                                    do_std = st.radio("Would you like to standardize the representation?", ["Yes", "No"], key=f"do_std_{issue_id}", index=None)
                                
                                    if do_std == "No":
                                        st.info("Left unchanged.")
                                        continue
                                    elif do_std != "Yes":
                                        continue
                                else:
                                    st.write("#### AI Suggestion")
                                    st.write(f"**Operation:** {suggestion.get('operation')}")
                                    if suggestion.get('column'):
                                        st.write(f"**Column:** {suggestion.get('column')}")
                                    if suggestion.get('parameters'):
                                        st.write(f"**Parameters:** {suggestion.get('parameters')}")
                            
                            if issue_id not in st.session_state.previews:
                                if st.button("Preview Changes", key=f"preview_btn_{issue_id}"):
                                    try:
                                        request = {
                                            "operation": suggestion["operation"],
                                            "column": suggestion.get("column"),
                                            "parameters": suggestion.get("parameters", {})
                                        }
                                        _, changes = execute_operation(working_df.copy(), request)
                                        col = suggestion.get("column")
                                        col_fingerprint = hashlib.md5(pd.util.hash_pandas_object(working_df[col]).values).hexdigest() if col in working_df.columns else None
                                        st.session_state.previews[issue_id] = {
                                            "changes": changes,
                                            "fingerprint": col_fingerprint
                                        }
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error previewing changes: {e}")
                            else:
                                preview_data = st.session_state.previews[issue_id]
                                col = suggestion.get("column")
                                current_fingerprint = hashlib.md5(pd.util.hash_pandas_object(working_df[col]).values).hexdigest() if col and col in working_df.columns else None
                                
                                if preview_data.get("fingerprint") != current_fingerprint:
                                    st.warning("Data has changed since this preview was generated. Please regenerate it.")
                                    if st.button("Regenerate Preview", key=f"regen_btn_{issue_id}"):
                                        del st.session_state.previews[issue_id]
                                        st.rerun()
                                    continue
                                
                                changes = preview_data["changes"]
                                st.write("#### Preview Changes")
                                if not changes:
                                    st.info("No changes would be made.")
                                else:
                                    st.write(f"**Total changes:** {len(changes)}")
                                    st.dataframe(pd.DataFrame(changes))
                                
                                st.info("No data was modified. This is just a preview.")
                                
                                if st.button("Apply Suggestion", key=f"apply_btn_{issue_id}"):
                                    try:
                                        request = {
                                            "operation": suggestion["operation"],
                                            "column": suggestion.get("column"),
                                            "parameters": suggestion.get("parameters", {})
                                        }
                                        cleaned_df, applied_changes = execute_operation(working_df.copy(), request)
                                        st.session_state.audit_log.extend(applied_changes)
                                        st.session_state.working_df = cleaned_df
                                        
                                        st.session_state.resolved_issues.add(issue_id)
                                        
                                        # Selectively invalidate state for the applied column
                                        applied_col = request.get("column")
                                        if applied_col:
                                            for k in list(st.session_state.suggestions.keys()):
                                                if st.session_state.suggestions[k].get("column") == applied_col and k != issue_id:
                                                    if k in st.session_state.previews:
                                                        del st.session_state.previews[k]
                                                    # Note: We don't necessarily delete the suggestion itself anymore,
                                                    # because the re-evaluation logic at the top will handle it automatically!
                                        
                                        if issue_id in st.session_state.previews:
                                            del st.session_state.previews[issue_id]
                                            
                                        st.success("Suggestion applied successfully!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error applying changes: {e}")

        if st.session_state.audit_log:
            st.divider()
            st.header("Audit Trail")
            with st.expander(f"View applied changes ({len(st.session_state.audit_log)})", expanded=False):
                st.dataframe(pd.DataFrame(st.session_state.audit_log))

if __name__ == "__main__":
    main()

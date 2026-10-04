import pandas as pd
import re

def resolve_suggestion(suggestion, issue, df):
    issue_type = issue.get("type")
    column = issue.get("column")
    
    ai_op = suggestion.get("operation")
    ai_params = suggestion.get("parameters", {})
    
    # Check if a column is clearly numeric or clearly categorical
    col_classification = "ambiguous"
    if column is not None and column in df.columns:
        non_nulls = df[column].dropna()
        if non_nulls.empty:
            pass
        elif pd.api.types.is_numeric_dtype(df[column]):
            col_classification = "numeric"
        else:
            numeric_coerced = pd.to_numeric(non_nulls, errors='coerce')
            numeric_count = numeric_coerced.notna().sum()
            if numeric_count == len(non_nulls):
                col_classification = "numeric"
            elif numeric_count == 0:
                col_classification = "categorical"
            
    # 1. Missing values
    if issue_type == "missing_values":
        if col_classification == "numeric":
            return {
                "operation": "fill_numeric",
                "column": column,
                "parameters": ai_params if ai_params else {"strategy": "median"}
            }
        elif col_classification == "categorical":
            return {
                "operation": "fill_categorical",
                "column": column,
                "parameters": ai_params if ai_params else {"strategy": "mode"}
            }
        else:
            from unit_registry import normalize_unit
            non_nulls = df[column].dropna().astype(str).str.strip()
            num_total = len(non_nulls)
            unit_pattern = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*([a-zA-Z]+)\s*$")
            num_with_units = 0
            pure_numeric_count = 0
            units_found = set()
            
            for val in non_nulls:
                match = unit_pattern.match(val)
                if match:
                    num_with_units += 1
                    units_found.add(normalize_unit(match.group(1)))
                else:
                    try:
                        float(val)
                        pure_numeric_count += 1
                    except ValueError:
                        pass
                        
            if 0 < num_with_units < num_total and pure_numeric_count == (num_total - num_with_units) and len(units_found) == 1:
                return {
                    "resolution_type": "MIXED_NUMERIC",
                    "operation": "remove_units",
                    "column": column,
                    "parameters": {"unit": list(units_found)[0]},
                    "message": f"Mixed numeric representation detected. Some values contain '{list(units_found)[0]}' while others are purely numeric. Standardize numeric representation first?"
                }
                
            return "Review required: Column contains mixed numeric and text values. Safe resolution not possible."
    # 3. Exact duplicates
    if issue_type == "duplicate_rows":
        if ai_op and ai_op != "remove_duplicates":
            return "Review required: AI suggestion mismatch for duplicate rows."
        return {
            "operation": "remove_duplicates",
            "column": None,
            "parameters": {}
        }
        
    # 4. Text inconsistency
    if issue_type == "text_inconsistency":
        variants = issue.get("variants", [])
        
        lowered_groups = {}
        for v in variants:
            key = v.strip().lower()
            lowered_groups.setdefault(key, set()).add(v)
            
        for key, group in lowered_groups.items():
            if len(group) <= 1:
                return "Review required: Semantic text differences detected. Cannot be safely standardized automatically."
                
        stripped = set(v.strip() for v in variants)
        lowered = set(v.strip().lower() for v in variants)
            
        if len(stripped) == len(lowered):
            return {
                "operation": "standardize_text",
                "column": column,
                "parameters": {"style": "strip"}
            }
            
        if ai_op == "standardize_text" and "style" in ai_params:
            style = ai_params["style"]
            if style in ["strip", "lower", "upper", "title"]:
                return {
                    "operation": "standardize_text",
                    "column": column,
                    "parameters": {"style": style}
                }
                
        # Fallback to dominant style if AI didn't provide one
        styles = {"title": 0, "lower": 0, "upper": 0}
        for val in df[column].dropna():
            val_str = str(val).strip()
            if not any(c.isalpha() for c in val_str):
                continue
            if val_str == val_str.title():
                styles["title"] += 1
            elif val_str == val_str.lower():
                styles["lower"] += 1
            elif val_str == val_str.upper():
                styles["upper"] += 1
                
        total_counted = sum(styles.values())
        if total_counted > 0:
            best_style = max(styles, key=styles.get)
            
            # Sort the counts to find the highest and second highest
            counts = sorted(styles.values(), reverse=True)
            highest_count = counts[0]
            second_highest_count = counts[1] if len(counts) > 1 else 0
            
            # Safe if it's at least 50% and strictly greater than any other style
            if highest_count >= total_counted * 0.5 and highest_count > second_highest_count:
                return {
                    "operation": "standardize_text",
                    "column": column,
                    "parameters": {"style": best_style}
                }
                
        return "Review required: Text casing inconsistencies detected, AI could not determine a safe style."

    # 5. Embedded units
    if issue_type == "embedded_units":
        from unit_registry import normalize_unit, get_unit_category
        
        non_nulls = df[column].dropna().astype(str).str.strip()
        num_total = len(non_nulls)
        
        unit_pattern = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*([a-zA-Z]+)\s*$")
        
        num_with_units = 0
        pure_numeric_count = 0
        units_found = set()
        
        for val in non_nulls:
            match = unit_pattern.match(val)
            if match:
                num_with_units += 1
                units_found.add(normalize_unit(match.group(1)))
            else:
                try:
                    float(val)
                    pure_numeric_count += 1
                except ValueError:
                    pass
        
        if num_with_units < num_total:
            if pure_numeric_count == (num_total - num_with_units):
                if len(units_found) == 1:
                    if df[column].isna().any():
                        return {
                            "resolution_type": "REVIEW_REQUIRED",
                            "message": "Mixed numeric representation will be resolved as part of the missing-value issue."
                        }
                    return {
                        "resolution_type": "MIXED_NUMERIC",
                        "operation": "remove_units",
                        "column": column,
                        "parameters": {"unit": list(units_found)[0]},
                        "message": f"Mixed numeric representation: Some values contain '{list(units_found)[0]}' while others are purely numeric."
                    }
                else:
                    return "Review required: Mixed numeric representation with multiple inconsistent units."
            else:
                return "Review required: Column contains inconsistent text values mixed with numeric values."
                
        if len(units_found) > 1:
            categories = {get_unit_category(u) for u in units_found}
            if "Unknown" in categories or len(categories) > 1:
                cat_list = ", ".join(sorted(categories))
                unit_list = ", ".join(sorted(units_found))
                return f"Review required: Genuine unit inconsistency. Detected units ({unit_list}) belong to conflicting categories ({cat_list})."
            else:
                category = list(categories)[0]
                from unit_registry import UNIT_REGISTRY
                available_targets = UNIT_REGISTRY[category]
                return {
                    "resolution_type": "STANDARDIZATION_AVAILABLE",
                    "operation": "standardize_units",
                    "column": column,
                    "available_targets": available_targets,
                    "detected_units": list(units_found)
                }
                
        if len(units_found) == 1:
            if ai_op and (ai_op != "remove_units" or ai_params.get("unit") != list(units_found)[0]):
                return "Review required: AI suggestion mismatch with deterministic unit evidence."
            return {
                "operation": "remove_units",
                "column": column,
                "parameters": {"unit": list(units_found)[0]}
            }
            
        return "Review required: Could not safely extract unit."

    # 6. Date format inconsistency
    if issue_type == "date_format_inconsistency":
        return "Review required: Multiple date formats detected. Target format must be specified by the user."

    # If it's something else not caught above, return review required
    return "Review required: Safe automatic resolution not implemented for this issue."

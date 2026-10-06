from pydantic import BaseModel
from typing import Any
import pandas as pd
import math

class ColumnContext(BaseModel):
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    semantic_hint: str | None = None
    
    min_val: float | None = None
    max_val: float | None = None
    
    top_categories: list[Any] | None = None
    
    date_min: str | None = None
    date_max: str | None = None
    
    semantic_role: str = "unknown"
    semantic_confidence: str = "low"
    semantic_evidence: str | None = None

class DatasetContext(BaseModel):
    row_count: int
    column_count: int
    columns: list[ColumnContext]
    sample_rows: list[dict[str, Any]]

class ContextBuilder:
    def _infer_semantic_role(self, col_name: str, s: pd.Series, unique_count: int, row_count: int, min_val: float | None, max_val: float | None, is_date: bool) -> tuple[str, str, str]:
        name_lower = col_name.lower()
        non_null = s.dropna()
        if len(non_null) == 0:
            return "unknown", "low", "Column is entirely empty"
            
        is_str_or_obj = pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
        is_numeric = pd.api.types.is_numeric_dtype(s)
        
        # 1. Identifier (Strong)
        is_id_name = any(x in name_lower for x in ['id', 'uuid', 'guid', 'index'])
        if unique_count == row_count and row_count > 1 and is_id_name:
            return "identifier", "high", "Unique values across all rows and column name suggests ID"
                
        # 2. Date
        if is_date:
            if any(x in name_lower for x in ['date', 'time', 'timestamp', 'created', 'updated']):
                return "date/datetime", "high", "Parsed as dates and name suggests temporal data"
            return "date/datetime", "medium", "Values successfully parsed as dates"
            
        # 3. Email
        if is_str_or_obj:
            emails = non_null.astype(str).str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$')
            if emails.mean() > 0.9:
                return "email", "high", ">90% of values match email format"
                
        # 4. Phone
        if is_str_or_obj:
            is_phone_name = any(x in name_lower for x in ['phone', 'mobile', 'tel', 'contact'])
            phones = non_null.astype(str).str.match(r'^\+?[\d\s\-\(\)]{7,20}$')
            if phones.mean() > 0.8:
                if is_phone_name:
                    return "phone", "high", ">80% match phone format and name suggests phone"
                return "phone", "medium", ">80% match phone format"
                
        # 5. Age
        if is_numeric:
            is_age_name = any(x in name_lower for x in ['age', 'years'])
            if is_age_name and min_val is not None and min_val >= 0 and max_val is not None and max_val <= 120:
                return "age", "high", "Column name suggests age and values are in typical 0-120 range"
            if is_age_name:
                return "age", "medium", "Column name suggests age but range is atypical"
                
        # 6. Monetary/Amount
        if is_numeric:
            is_money_name = any(x in name_lower for x in ['price', 'cost', 'revenue', 'salary', 'amount', 'total', 'tax', 'fee', 'balance'])
            if is_money_name:
                return "monetary/numeric amount", "high", "Numeric column with name suggesting financial/quantitative amount"
                
        # 7. Percentage
        if is_numeric:
            is_pct_name = any(x in name_lower for x in ['percent', 'rate', 'ratio', 'margin'])
            if is_pct_name:
                return "percentage", "high", "Numeric column with name suggesting percentage/rate"
                
        # 8. Name/Text (Do before categorical)
        if is_str_or_obj:
            is_text_name = any(x in name_lower for x in ['name', 'title', 'description', 'text', 'comment', 'address'])
            if is_text_name:
                return "name/text", "high", "String column with name suggesting text or entity name"

        # 9. Boolean/Binary
        if unique_count == 2:
            is_bool_name = any(x in name_lower for x in ['is_', 'has_', 'flag'])
            is_numeric_bool = is_numeric and set(non_null.dropna().unique()).issubset({0, 1, 0.0, 1.0})
            is_str_bool = is_str_or_obj and set(str(x).lower() for x in non_null.dropna().unique()).issubset({'true', 'false', 'yes', 'no', 'y', 'n', 't', 'f'})
            if is_bool_name or is_numeric_bool or is_str_bool:
                return "boolean/binary", "high", "Exactly two unique values with binary naming or typical boolean values"
            
        # 10. Categorical
        if 0 < unique_count < 20 and unique_count < row_count:
            is_cat_name = any(x in name_lower for x in ['category', 'type', 'status', 'group', 'role', 'city', 'country', 'state', 'gender', 'department', 'class'])
            if is_cat_name:
                return "categorical", "high", f"Column name implies category and has low cardinality ({unique_count})"
                
        # 11. Identifier (Weak fallback)
        is_integer = pd.api.types.is_integer_dtype(s)
        if unique_count == row_count and row_count > 1 and (is_integer or is_str_or_obj):
            return "identifier", "medium", "Values are completely unique across all rows"
                
        # Default for remaining numerics
        if is_numeric:
            return "numeric", "low", "Numeric column but no specific role identified"
            
        return "unknown", "low", "Insufficient evidence to determine specific semantic role"

    def build_context(self, df: pd.DataFrame) -> DatasetContext:
        columns = []
        row_count = len(df)
        for col in df.columns:
            s = df[col]
            missing_count = int(s.isna().sum())
            missing_percentage = float(missing_count / row_count) if row_count > 0 else 0.0
            
            non_null_s = s.dropna()
            unique_count = int(non_null_s.nunique())
            
            dtype_str = str(s.dtype)
            hint = None
            
            min_val = max_val = top_cats = date_min = date_max = None
            
            # Semantic Hinting
            if unique_count == row_count and row_count > 0 and pd.api.types.is_numeric_dtype(s):
                hint = "Likely Identifier"
                
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                if unique_count > 0:
                    try:
                        # Ensure standard python floats to be JSON serializable
                        min_v = float(non_null_s.min())
                        max_v = float(non_null_s.max())
                        if not math.isnan(min_v) and not math.isinf(min_v):
                            min_val = min_v
                        if not math.isnan(max_v) and not math.isinf(max_v):
                            max_val = max_v
                    except:
                        pass
            
            if pd.api.types.is_datetime64_any_dtype(s):
                hint = "Likely Date"
                valid_dates = s.dropna()
                if len(valid_dates) > 0:
                    date_min = str(valid_dates.min())
                    date_max = str(valid_dates.max())
            elif pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
                # Date detection (conservative)
                if len(non_null_s) > 0:
                    try:
                        parsed_dates = pd.to_datetime(non_null_s, errors='coerce', format='mixed')
                        date_ratio = parsed_dates.notna().mean()
                        if date_ratio > 0.95:  # Conservative
                            hint = "Likely Date"
                            valid_dates = parsed_dates.dropna()
                            date_min = str(valid_dates.min())
                            date_max = str(valid_dates.max())
                    except:
                        pass
                
                # Categorical bounding (Max 10)
                if hint != "Likely Date" and 0 < unique_count < 20:
                    hint = "Likely Categorical"
                    val_counts = non_null_s.value_counts().head(10)
                    top_cats = [str(x) for x in val_counts.index.tolist()]
            
            if pd.api.types.is_bool_dtype(s):
                hint = "Likely Boolean"

            is_date = (date_min is not None)
            role, conf, evidence = self._infer_semantic_role(
                str(col), s, unique_count, row_count, min_val, max_val, is_date
            )

            columns.append(ColumnContext(
                name=str(col), dtype=dtype_str, missing_count=missing_count,
                missing_percentage=missing_percentage, unique_count=unique_count,
                semantic_hint=hint, min_val=min_val, max_val=max_val,
                top_categories=top_cats, date_min=date_min, date_max=date_max,
                semantic_role=role, semantic_confidence=conf, semantic_evidence=evidence
            ))
            
        sample = df.head(3).where(pd.notnull(df), None).to_dict(orient='records')
        clean_sample = []
        for row in sample:
            clean_row = {}
            for k, v in row.items():
                if pd.isna(v) or v is None:
                    clean_row[k] = None
                elif isinstance(v, (int, float, bool, str)):
                    if isinstance(v, float) and math.isnan(v):
                        clean_row[k] = None
                    else:
                        clean_row[k] = v
                else:
                    clean_row[k] = str(v)
            clean_sample.append(clean_row)
            
        return DatasetContext(
            row_count=row_count, column_count=len(df.columns),
            columns=columns, sample_rows=clean_sample
        )

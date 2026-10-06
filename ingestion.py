import csv
import io
from pydantic import BaseModel, Field
from typing import Literal, Any
import pandas as pd

class StructuralError(BaseModel):
    row_index: int | None = None
    expected_fields: int | None = None
    found_fields: int | None = None
    raw_content: str | None = None
    issue_type: Literal["extra_fields", "missing_fields", "malformed_quotes", "encoding_error", "empty_file", "other"]
    message: str | None = None

class IngestionResult(BaseModel):
    status: Literal["success", "needs_clarification", "fatal_error"]
    message: str | None = None
    dataframe: Any | None = None 
    structural_errors: list[StructuralError] | None = None
    
    class Config:
        arbitrary_types_allowed = True

class DataIngester:
    def ingest_csv(self, file_path: str, delimiter: str = ',') -> IngestionResult:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                content = f.read()
        except UnicodeDecodeError as e:
            return IngestionResult(
                status="fatal_error",
                message="File encoding is not valid UTF-8. Lossy decoding is rejected.",
                structural_errors=[StructuralError(issue_type="encoding_error", message=str(e))]
            )
        
        if not content.strip():
            return IngestionResult(
                status="fatal_error",
                message="File is empty.",
                structural_errors=[StructuralError(issue_type="empty_file")]
            )

        reader = csv.reader(io.StringIO(content), delimiter=delimiter, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            return IngestionResult(
                status="fatal_error",
                message="File has no header.",
                structural_errors=[StructuralError(issue_type="empty_file")]
            )
        except csv.Error as e:
            return IngestionResult(
                status="fatal_error",
                message="CSV parsing error in header.",
                structural_errors=[StructuralError(issue_type="malformed_quotes", message=str(e))]
            )
            
        expected_fields = len(header)
        errors = []
        valid_rows = []
        
        row_idx = 1
        while True:
            row_idx += 1
            try:
                row = next(reader)
                if len(row) > expected_fields:
                    errors.append(StructuralError(
                        row_index=row_idx, 
                        expected_fields=expected_fields, 
                        found_fields=len(row), 
                        issue_type="extra_fields"
                    ))
                elif len(row) < expected_fields:
                    errors.append(StructuralError(
                        row_index=row_idx, 
                        expected_fields=expected_fields, 
                        found_fields=len(row), 
                        issue_type="missing_fields"
                    ))
                else:
                    valid_rows.append(row)
            except StopIteration:
                break
            except csv.Error as e:
                errors.append(StructuralError(
                    row_index=row_idx, 
                    issue_type="malformed_quotes", 
                    message=str(e)
                ))
                
        if errors:
            return IngestionResult(
                status="needs_clarification", 
                message=f"Found {len(errors)} structural issues. Safe repair is ambiguous.", 
                structural_errors=errors
            )
            
        df = pd.DataFrame(valid_rows, columns=header)
        df = self._safe_infer_types(df)
        return IngestionResult(status="success", dataframe=df)

    def _safe_infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            s = df[col]
            if not pd.api.types.is_object_dtype(s) and not pd.api.types.is_string_dtype(s):
                continue
                
            non_null = s.dropna().astype(str).str.strip()
            if len(non_null) == 0:
                continue
                
            # Try Numeric
            if not non_null.str.match(r'^-?0[0-9]+$').any():
                converted = pd.to_numeric(s, errors='coerce')
                is_originally_null = s.isna() | s.astype(str).str.strip().isin(['', 'nan', 'NaN', 'None', 'null'])
                is_now_null = converted.isna()
                
                if not (is_now_null & ~is_originally_null).any():
                    df[col] = converted
                    continue
                    
            # Try Datetime
            if (non_null.str.len() >= 6).all():
                try:
                    converted_date = pd.to_datetime(s, errors='coerce', format='mixed')
                    is_now_null = converted_date.isna()
                    is_originally_null = s.isna() | s.astype(str).str.strip().isin(['', 'nan', 'NaN', 'None', 'null'])
                    
                    if not (is_now_null & ~is_originally_null).any():
                        df[col] = converted_date
                except Exception:
                    pass
                    
        return df

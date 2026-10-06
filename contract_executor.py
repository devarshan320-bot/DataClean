import ast
import operator
import pandas as pd
import numpy as np
import re
from contracts import ReadyContract, Operation, Condition
from contract_validator import preprocess_expression

class SafeASTEvaluator(ast.NodeVisitor):
    def __init__(self, df: pd.DataFrame, mappings: dict):
        self.df = df
        self.mappings = mappings
        self.ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos
        }

    def evaluate(self, node):
        if isinstance(node, ast.Expression):
            return self.evaluate(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            actual_col = self.mappings.get(node.id, node.id)
            return self.df[actual_col]
        elif isinstance(node, ast.BinOp):
            return self.ops[type(node.op)](self.evaluate(node.left), self.evaluate(node.right))
        elif isinstance(node, ast.UnaryOp):
            return self.ops[type(node.op)](self.evaluate(node.operand))
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")

def evaluate_condition(df: pd.DataFrame, condition: Condition) -> pd.Series:
    if condition.operator == "and":
        mask = np.ones(len(df), dtype=bool)
        for c in condition.conditions:
            mask = mask & evaluate_condition(df, c)
        return mask
    elif condition.operator == "or":
        mask = np.zeros(len(df), dtype=bool)
        for c in condition.conditions:
            mask = mask | evaluate_condition(df, c)
        return mask
    elif condition.operator == "not":
        return ~evaluate_condition(df, condition.conditions[0])
            
    col = condition.column
    val = condition.value
    
    if condition.operator in ["contains", "starts_with", "ends_with"]:
        if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
            raise TypeError(f"String operator '{condition.operator}' cannot be applied to non-string column '{col}'.")
        
        # Explicitly handle mixed object types safely
        def check_str(x, op, target):
            if not isinstance(x, str): return False
            if op == "contains": return target.lower() in x.lower()
            if op == "starts_with": return x.startswith(target)
            if op == "ends_with": return x.endswith(target)
        
        return df[col].apply(lambda x: check_str(x, condition.operator, str(val)))

    try:
        # Note: Comparison behavior relies on Pandas. It does NOT perform implicit type conversion.
        # e.g., comparing an integer column to a string value will either yield False or raise TypeError.
        if condition.operator == "equals": return df[col] == val
        elif condition.operator == "not_equals": return df[col] != val
        elif condition.operator == "gt": return df[col] > val
        elif condition.operator == "gte": return df[col] >= val
        elif condition.operator == "lt": return df[col] < val
        elif condition.operator == "lte": return df[col] <= val
        elif condition.operator == "is_null": return df[col].isna()
        elif condition.operator == "is_not_null": return df[col].notna()
        elif condition.operator == "in": return df[col].isin(val)
        elif condition.operator == "not_in": return ~df[col].isin(val)
        elif condition.operator == "between": return df[col].between(val[0], val[1])
    except TypeError as e:
        raise TypeError(f"Type mismatch applying '{condition.operator}' to column '{col}': {e}")

def execute_operation(df: pd.DataFrame, op: Operation) -> pd.DataFrame:
    if op.op == "filter":
        return df[evaluate_condition(df, op.condition)].copy()
    elif op.op == "sort":
        asc = op.ascending if op.ascending is not None else [True] * len(op.by)
        return df.sort_values(by=op.by, ascending=asc)
    elif op.op == "drop_duplicates":
        return df.drop_duplicates(subset=op.subset)
    elif op.op == "select":
        return df[op.columns].copy()
    elif op.op == "drop":
        return df.drop(columns=op.columns)
    elif op.op == "mutate":
        safe_expr, mappings = preprocess_expression(op.expression.expr)
        tree = ast.parse(safe_expr, mode='eval')
        evaluator = SafeASTEvaluator(df, mappings)
        df = df.copy()
        df[op.column] = evaluator.evaluate(tree.body)
        return df
    elif op.op == "normalize_text":
        df = df.copy()
        cols = op.columns if isinstance(op.columns, list) else df.select_dtypes(include=['object', 'string']).columns
        for col in cols:
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                raise TypeError(f"normalize_text cannot be applied to non-string column '{col}'.")
            
            def apply_style(x, style):
                if not isinstance(x, str): return x
                if style == "strip": return x.strip()
                elif style == "collapse_whitespace": return re.sub(r'\s+', ' ', x).strip()
                elif style == "lower": return x.lower()
                elif style == "upper": return x.upper()
                elif style == "title": return x.title()
                return x
            df[col] = df[col].apply(lambda x: apply_style(x, op.style))
        return df
    return df

def execute_contract(df: pd.DataFrame, contract: ReadyContract) -> tuple[pd.DataFrame, dict]:
    rows_before = len(df)
    cols_before = list(df.columns)
    
    current_df = df.copy()
    for op in contract.operations:
        current_df = execute_operation(current_df, op)
        
    cols_after = list(current_df.columns)
    
    common_cols = [c for c in cols_before if c in cols_after]
    values_changed = 0
    if common_cols and not current_df.empty and not df.empty:
        df_old_common = df[common_cols]
        df_new_common = current_df[common_cols]
        
        common_idx = df_old_common.index.intersection(df_new_common.index)
        
        if not common_idx.empty:
            df_old_aligned = df_old_common.loc[common_idx]
            df_new_aligned = df_new_common.loc[common_idx]
            
            ne = df_old_aligned != df_new_aligned
            both_na = df_old_aligned.isna() & df_new_aligned.isna()
            values_changed = int((ne & ~both_na).sum().sum())

    impact = {
        "Rows before": rows_before,
        "Rows after": len(current_df),
        "Rows removed": max(0, rows_before - len(current_df)),
        "Columns added": [c for c in cols_after if c not in cols_before],
        "Columns removed": [c for c in cols_before if c not in cols_after],
        "Values changed": values_changed
    }
    return current_df, impact

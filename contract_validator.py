import ast
import re
from contracts import ReadyContract, Condition

def preprocess_expression(expr: str) -> tuple[str, dict]:
    mappings = {}
    def replacer(match):
        col_name = match.group(1)
        safe_name = f"__col_{len(mappings)}__"
        mappings[safe_name] = col_name
        return safe_name
    safe_expr = re.sub(r'`([^`]+)`', replacer, expr)
    return safe_expr, mappings

class ASTValidator(ast.NodeVisitor):
    def __init__(self, allowed_columns, mappings):
        self.allowed_columns = allowed_columns
        self.mappings = mappings
        self.errors = []
        
    def generic_visit(self, node):
        allowed = (
            ast.Module, ast.Expr, ast.Expression, ast.BinOp, ast.UnaryOp, 
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, 
            ast.UAdd, ast.USub, ast.Name, ast.Constant, ast.Load
        )
        if not isinstance(node, allowed):
            self.errors.append(f"Forbidden syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        actual_col = self.mappings.get(node.id, node.id)
        if actual_col not in self.allowed_columns:
            self.errors.append(f"Invalid column reference in expression: '{actual_col}'")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, float)):
            self.errors.append(f"Invalid constant type '{type(node.value).__name__}'. Only numeric constants are supported.")
        self.generic_visit(node)

def validate_expression(expr_str: str, columns: list[str]) -> list[str]:
    safe_expr, mappings = preprocess_expression(expr_str)
    try:
        tree = ast.parse(safe_expr, mode='eval')
    except SyntaxError as e:
        return [f"Syntax error in expression: {e}"]
        
    validator = ASTValidator(columns, mappings)
    validator.visit(tree)
    return validator.errors

def validate_condition(condition: Condition, columns: list[str]) -> list[str]:
    errors = []
    if condition.operator in ["and", "or", "not"]:
        for c in (condition.conditions or []):
            errors.extend(validate_condition(c, columns))
    else:
        if condition.column not in columns:
            errors.append(f"Column '{condition.column}' not found.")
    return errors

def validate_contract(contract: ReadyContract, columns: list[str]) -> list[str]:
    errors = []
    current_cols = list(columns)
    
    for op in contract.operations:
        if op.op == "filter":
            errors.extend(validate_condition(op.condition, current_cols))
        elif op.op == "sort":
            for col in op.by:
                if col not in current_cols:
                    errors.append(f"Sort column '{col}' not found.")
        elif op.op == "mutate":
            errors.extend(validate_expression(op.expression.expr, current_cols))
            if op.column not in current_cols:
                current_cols.append(op.column)
        elif op.op in ("select", "drop"):
            for col in op.columns:
                if col not in current_cols:
                    errors.append(f"Column '{col}' not found for {op.op}.")
            if op.op == "select" and not errors:
                current_cols = op.columns
            elif op.op == "drop" and not errors:
                current_cols = [c for c in current_cols if c not in op.columns]
        elif op.op == "normalize_text":
            if isinstance(op.columns, list):
                for col in op.columns:
                    if col not in current_cols:
                        errors.append(f"Normalize column '{col}' not found.")
    return errors

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Any, Union, Annotated

class Condition(BaseModel):
    operator: Literal[
        "equals", "not_equals", "gt", "gte", "lt", "lte",
        "contains", "starts_with", "ends_with",
        "is_null", "is_not_null", "in", "not_in", "between",
        "and", "or", "not"
    ]
    column: str | None = None
    value: Any | None = None
    conditions: list['Condition'] | None = None

    @model_validator(mode='after')
    def validate_structure(self):
        is_logical = self.operator in ["and", "or", "not"]
        if is_logical:
            if self.column is not None or self.value is not None:
                raise ValueError(f"Logical operator '{self.operator}' cannot have 'column' or 'value'.")
            if not self.conditions:
                raise ValueError(f"Logical operator '{self.operator}' requires 'conditions' list.")
            if self.operator == "not" and len(self.conditions) != 1:
                raise ValueError("Operator 'not' requires exactly one condition.")
        else:
            if self.conditions is not None:
                raise ValueError(f"Leaf operator '{self.operator}' cannot have 'conditions'.")
            if self.column is None:
                raise ValueError(f"Operator '{self.operator}' requires a 'column'.")
            
            if self.operator in ["is_null", "is_not_null"]:
                if self.value is not None:
                    raise ValueError(f"Operator '{self.operator}' must not have a value.")
            elif self.operator in ["in", "not_in"]:
                if not isinstance(self.value, (list, tuple)):
                    raise ValueError(f"Operator '{self.operator}' requires a list-like value.")
            elif self.operator == "between":
                if not isinstance(self.value, (list, tuple)) or len(self.value) != 2:
                    raise ValueError("Operator 'between' requires exactly two values.")
            else:
                if self.value is None:
                    raise ValueError(f"Operator '{self.operator}' requires a value.")
        return self

class Expression(BaseModel):
    expr: str = Field(description="Mathematical expression")

class FilterOp(BaseModel):
    op: Literal["filter"] = "filter"
    condition: Condition

class SortOp(BaseModel):
    op: Literal["sort"] = "sort"
    by: list[str]
    ascending: list[bool] | None = None
    
    @model_validator(mode='after')
    def validate_ascending_length(self):
        if self.ascending is not None and len(self.ascending) != len(self.by):
            raise ValueError("Length of 'ascending' must match length of 'by'.")
        return self

class DropDuplicatesOp(BaseModel):
    op: Literal["drop_duplicates"] = "drop_duplicates"
    subset: list[str] | None = None

class MutateOp(BaseModel):
    op: Literal["mutate"] = "mutate"
    column: str
    expression: Expression

class NormalizeTextOp(BaseModel):
    op: Literal["normalize_text"] = "normalize_text"
    style: Literal["strip", "collapse_whitespace", "lower", "upper", "title"]
    columns: list[str] | Literal["all_string"] | None = None

class SelectOp(BaseModel):
    op: Literal["select"] = "select"
    columns: list[str]

class DropOp(BaseModel):
    op: Literal["drop"] = "drop"
    columns: list[str]

Operation = Annotated[
    Union[FilterOp, SortOp, DropDuplicatesOp, MutateOp, NormalizeTextOp, SelectOp, DropOp],
    Field(discriminator='op')
]

class BaseContract(BaseModel):
    status: Literal["ready", "needs_clarification", "unsupported", "invalid"]
    message: str | None = None

class ReadyContract(BaseContract):
    status: Literal["ready"] = "ready"
    operations: list[Operation]

class ClarificationContract(BaseContract):
    status: Literal["needs_clarification"] = "needs_clarification"
    options: list[str] | None = None

class UnsupportedContract(BaseContract):
    status: Literal["unsupported"] = "unsupported"

class InvalidContract(BaseContract):
    status: Literal["invalid"] = "invalid"
    errors: list[str]

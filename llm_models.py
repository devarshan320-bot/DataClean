from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from ir_models import (
    Reference,
    ComparisonOp,
    Value,
    LogicalOp,
    IntentAction,
    FilterMode
)

class FlatCondition(BaseModel):
    subject: Reference
    operator: ComparisonOp
    value: Optional[Value] = None

class FlatStep(BaseModel):
    intent: IntentAction
    target_entity: Optional[str] = Field(None, description="e.g. 'customers', 'records', 'everyone', null if absent")
    
    # FILTER
    filter_mode: Optional[FilterMode] = None
    conditions: Optional[List[FlatCondition]] = None
    logic: Optional[LogicalOp] = None
    
    # SORT
    sort_subject: Optional[Reference] = None
    sort_direction: Optional[Literal["ascending", "descending"]] = None
    
    # MUTATE
    mutate_subject: Optional[Reference] = None
    mutation_operation: Optional[str] = None
    
    # AGGREGATE
    aggregate_subject: Optional[Reference] = None
    aggregate_function: Optional[str] = None
    
    # DROP / SELECT
    columns_subject: Optional[List[Reference]] = None

class FlatInterpretationIR(BaseModel):
    steps: List[FlatStep]

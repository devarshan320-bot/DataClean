from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Any, Union, Annotated
from enum import Enum

class ReferenceType(str, Enum):
    EXPLICIT_COLUMN = "explicit_column"
    VAGUE_CONCEPT = "vague_concept"
    IMPLICIT_FROM_VALUE = "implicit_from_value"

class Reference(BaseModel):
    reference_type: ReferenceType
    original_text: Optional[str] = Field(..., description="The exact words the user typed, or null if implicit")

class SemanticValueType(str, Enum):
    NUMBER = "number"
    TEXT = "text"
    MONTH = "month"
    DATE = "date"
    BOOLEAN = "boolean"
    NULL = "null"

class Value(BaseModel):
    original_text: str
    semantic_type: SemanticValueType
    parsed_value: Any

class NodeType(str, Enum):
    LOGICAL = "logical"
    COMPARISON = "comparison"

class LogicalOp(str, Enum):
    AND = "and"
    OR = "or"
    NOT = "not"

class ComparisonOp(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUALS = "greater_than_or_equals"
    LESS_THAN_OR_EQUALS = "less_than_or_equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    TEMPORAL_MATCH = "temporal_match" 
    IS_NULL = "is_null"



class LogicalNode(BaseModel):
    node_type: Literal[NodeType.LOGICAL] = NodeType.LOGICAL
    logic_op: LogicalOp
    children: List['ConditionNode']

class ComparisonNode(BaseModel):
    node_type: Literal[NodeType.COMPARISON] = NodeType.COMPARISON
    subject: Reference
    operator: ComparisonOp
    value: Optional[Value] = None

ConditionNode = Annotated[Union[LogicalNode, ComparisonNode], Field(discriminator="node_type")]

class IntentAction(str, Enum):
    FILTER = "filter"
    SORT = "sort"
    DEDUPLICATE = "deduplicate"
    MUTATE = "mutate"
    AGGREGATE = "aggregate"  
    DROP = "drop"
    SELECT = "select"

class FilterMode(str, Enum):
    KEEP = "keep"
    REMOVE = "remove"

class Step(BaseModel):
    intent: IntentAction
    target_entity: Optional[str] = Field(None, description="e.g. 'customers', 'records', 'everyone', null if absent")
    
    # FILTER
    filter_mode: Optional[FilterMode] = None
    condition: Optional[ConditionNode] = None
    
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

class InterpretationIR(BaseModel):
    steps: List[Step]

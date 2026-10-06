import pytest
from ir_compiler import compile_flat_ir
from llm_models import FlatInterpretationIR, FlatStep, FlatCondition
from ir_models import (
    InterpretationIR, Reference, ReferenceType, Value, SemanticValueType, 
    IntentAction, FilterMode, ComparisonOp, LogicalOp, NodeType, 
    LogicalNode, ComparisonNode
)

def test_compile_flat_ir_a_deduplicate():
    # A: "Remove duplicates."
    flat_ir = FlatInterpretationIR(
        steps=[FlatStep(intent=IntentAction.DEDUPLICATE)]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 1
    assert ir.steps[0].intent == IntentAction.DEDUPLICATE

def test_compile_flat_ir_b_drop():
    # B: "Drop the Age column."
    ref = Reference(reference_type=ReferenceType.EXPLICIT_COLUMN, original_text="Age")
    flat_ir = FlatInterpretationIR(
        steps=[FlatStep(intent=IntentAction.DROP, columns_subject=[ref])]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 1
    assert ir.steps[0].intent == IntentAction.DROP
    assert ir.steps[0].columns_subject == [ref]

def test_compile_flat_ir_c_filter_single_condition():
    # C: "Drop any record where Age is below 18."
    ref = Reference(reference_type=ReferenceType.EXPLICIT_COLUMN, original_text="Age")
    val = Value(original_text="18", semantic_type=SemanticValueType.NUMBER, parsed_value=18)
    cond = FlatCondition(subject=ref, operator=ComparisonOp.LESS_THAN, value=val)
    
    flat_ir = FlatInterpretationIR(
        steps=[FlatStep(
            intent=IntentAction.FILTER, 
            filter_mode=FilterMode.REMOVE, 
            conditions=[cond]
        )]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 1
    step = ir.steps[0]
    assert step.intent == IntentAction.FILTER
    assert step.filter_mode == FilterMode.REMOVE
    
    # Must compile to single ComparisonNode
    assert isinstance(step.condition, ComparisonNode)
    assert step.condition.node_type == NodeType.COMPARISON
    assert step.condition.subject == ref
    assert step.condition.operator == ComparisonOp.LESS_THAN
    assert step.condition.value == val

def test_compile_flat_ir_d_filter_multiple_conditions():
    # D: "Keep Bangalore customers who spent more than 500."
    ref1 = Reference(reference_type=ReferenceType.IMPLICIT_FROM_VALUE, original_text="Bangalore")
    val1 = Value(original_text="Bangalore", semantic_type=SemanticValueType.TEXT, parsed_value="Bangalore")
    cond1 = FlatCondition(subject=ref1, operator=ComparisonOp.EQUALS, value=val1)
    
    ref2 = Reference(reference_type=ReferenceType.VAGUE_CONCEPT, original_text="spent")
    val2 = Value(original_text="500", semantic_type=SemanticValueType.NUMBER, parsed_value=500)
    cond2 = FlatCondition(subject=ref2, operator=ComparisonOp.GREATER_THAN, value=val2)
    
    flat_ir = FlatInterpretationIR(
        steps=[FlatStep(
            intent=IntentAction.FILTER, 
            filter_mode=FilterMode.KEEP, 
            conditions=[cond1, cond2],
            logic=LogicalOp.AND
        )]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 1
    step = ir.steps[0]
    assert step.intent == IntentAction.FILTER
    assert step.filter_mode == FilterMode.KEEP
    
    # Must compile to LogicalNode with 2 children
    assert isinstance(step.condition, LogicalNode)
    assert step.condition.node_type == NodeType.LOGICAL
    assert step.condition.logic_op == LogicalOp.AND
    assert len(step.condition.children) == 2
    assert step.condition.children[0].subject == ref1
    assert step.condition.children[1].subject == ref2

def test_compile_flat_ir_e_sort():
    # E: "Sort by Revenue highest to lowest."
    ref = Reference(reference_type=ReferenceType.VAGUE_CONCEPT, original_text="Revenue")
    flat_ir = FlatInterpretationIR(
        steps=[FlatStep(
            intent=IntentAction.SORT, 
            sort_subject=ref,
            sort_direction="descending"
        )]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 1
    assert ir.steps[0].intent == IntentAction.SORT
    assert ir.steps[0].sort_subject == ref
    assert ir.steps[0].sort_direction == "descending"

def test_compile_flat_ir_f_multi_step():
    # F: "Remove duplicates and order Revenue highest to lowest."
    ref = Reference(reference_type=ReferenceType.VAGUE_CONCEPT, original_text="Revenue")
    flat_ir = FlatInterpretationIR(
        steps=[
            FlatStep(intent=IntentAction.DEDUPLICATE),
            FlatStep(intent=IntentAction.SORT, sort_subject=ref, sort_direction="descending")
        ]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 2
    assert ir.steps[0].intent == IntentAction.DEDUPLICATE
    assert ir.steps[1].intent == IntentAction.SORT
    assert ir.steps[1].sort_subject == ref

def test_compile_flat_ir_g_multi_step_identical():
    # G: Same multi-step logic
    # "Get rid of identical rows and order the remaining data by Revenue highest to lowest."
    ref = Reference(reference_type=ReferenceType.VAGUE_CONCEPT, original_text="Revenue")
    flat_ir = FlatInterpretationIR(
        steps=[
            FlatStep(intent=IntentAction.DEDUPLICATE),
            FlatStep(intent=IntentAction.SORT, sort_subject=ref, sort_direction="descending")
        ]
    )
    ir = compile_flat_ir(flat_ir)
    assert len(ir.steps) == 2
    assert ir.steps[0].intent == IntentAction.DEDUPLICATE
    assert ir.steps[1].intent == IntentAction.SORT

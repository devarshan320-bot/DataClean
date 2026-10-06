from llm_models import FlatInterpretationIR, FlatStep
from ir_models import (
    InterpretationIR, 
    Step, 
    ComparisonNode, 
    LogicalNode, 
    NodeType,
    LogicalOp
)

def compile_flat_ir(flat_ir: FlatInterpretationIR) -> InterpretationIR:
    compiled_steps = []
    
    for flat_step in flat_ir.steps:
        # Convert condition(s) if present
        condition_node = None
        if flat_step.conditions:
            if len(flat_step.conditions) == 1:
                flat_cond = flat_step.conditions[0]
                condition_node = ComparisonNode(
                    node_type=NodeType.COMPARISON,
                    subject=flat_cond.subject,
                    operator=flat_cond.operator,
                    value=flat_cond.value
                )
            else:
                children = []
                for flat_cond in flat_step.conditions:
                    children.append(ComparisonNode(
                        node_type=NodeType.COMPARISON,
                        subject=flat_cond.subject,
                        operator=flat_cond.operator,
                        value=flat_cond.value
                    ))
                logic = flat_step.logic if flat_step.logic else LogicalOp.AND
                condition_node = LogicalNode(
                    node_type=NodeType.LOGICAL,
                    logic_op=logic,
                    children=children
                )
        
        step = Step(
            intent=flat_step.intent,
            target_entity=flat_step.target_entity,
            filter_mode=flat_step.filter_mode,
            condition=condition_node,
            sort_subject=flat_step.sort_subject,
            sort_direction=flat_step.sort_direction,
            mutate_subject=flat_step.mutate_subject,
            mutation_operation=flat_step.mutation_operation,
            aggregate_subject=flat_step.aggregate_subject,
            aggregate_function=flat_step.aggregate_function,
            columns_subject=flat_step.columns_subject
        )
        compiled_steps.append(step)
        
    return InterpretationIR(steps=compiled_steps)

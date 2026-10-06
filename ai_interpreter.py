import json
import pandas as pd
from pydantic import ValidationError, BaseModel
from typing import Literal, Optional
from contracts import (
    Operation, ReadyContract, FilterOp, Condition, SortOp, DropDuplicatesOp, 
    MutateOp, NormalizeTextOp, SelectOp, DropOp
)
from contract_validator import validate_contract
from context_builder import DatasetContext
from ai_provider import AIProvider
from ir_models import (
    InterpretationIR, Step, ReferenceType, Reference, SemanticValueType, Value, 
    NodeType, LogicalOp, ComparisonOp, IntentAction, FilterMode, ConditionNode
)
from llm_models import FlatInterpretationIR
from ir_compiler import compile_flat_ir

class InterpreterResult(BaseModel):
    status: Literal["ready", "needs_clarification", "unsupported", "invalid"]
    message: str | None = None
    contract: ReadyContract | None = None
    errors: list[str] | None = None

class ResolutionError(Exception):
    def __init__(self, message: str, status: str = "needs_clarification"):
        self.message = message
        self.status = status
        super().__init__(self.message)

class Stage2Resolver:
    def __init__(self, context: DatasetContext, df: pd.DataFrame):
        self.context = context
        self.df = df
        
    def resolve(self, ir: InterpretationIR) -> list[Operation]:
        operations = []
        current_df = self.df.copy()
        current_context = self.context
        
        from contract_executor import execute_contract
        from context_builder import ContextBuilder
        
        for step in ir.steps:
            # We temporarily swap self state so _resolve_step uses the current scratchpad
            # (or we could instantiate a new resolver, but mutating self for the duration of the loop is also fine. Let's just mutate self.)
            self.df = current_df
            self.context = current_context
            
            op = self._resolve_step(step)
            operations.append(op)
            
            temp_contract = ReadyContract(operations=[op])
            columns = [c.name for c in self.context.columns]
            
            val_errors = validate_contract(temp_contract, columns)
            if val_errors:
                raise ResolutionError(f"Step '{step.intent.value}' produced invalid operation: {val_errors}", status="invalid")
                
            try:
                current_df, _ = execute_contract(current_df, temp_contract)
            except Exception as e:
                raise ResolutionError(f"Step '{step.intent.value}' failed during resolution: {e}", status="invalid")
                
            current_context = ContextBuilder().build_context(current_df)
            
        return operations
        
    def _resolve_step(self, step: Step) -> Operation:
        if step.intent == IntentAction.FILTER:
            if not step.condition:
                raise ResolutionError("Filter intent missing condition.", status="invalid")
            resolved_cond = self._resolve_condition(step.condition)
            
            if step.filter_mode == FilterMode.REMOVE:
                resolved_cond = Condition(operator="not", conditions=[resolved_cond])
                
            return FilterOp(condition=resolved_cond)
            
        elif step.intent == IntentAction.SORT:
            if not step.sort_subject:
                raise ResolutionError("Sort intent missing subject.", status="invalid")
            col = self._resolve_reference(step.sort_subject)
            asc = [True] if step.sort_direction != "descending" else [False]
            return SortOp(by=[col], ascending=asc)
            
        elif step.intent == IntentAction.DEDUPLICATE:
            return DropDuplicatesOp()
            
        elif step.intent == IntentAction.MUTATE:
            if not step.mutate_subject:
                raise ResolutionError("Mutate intent missing subject.", status="invalid")
            col = self._resolve_reference(step.mutate_subject)
            supported_styles = ["strip", "collapse_whitespace", "lower", "upper", "title"]
            if step.mutation_operation in supported_styles:
                return NormalizeTextOp(style=step.mutation_operation, columns=[col]) # type: ignore
            else:
                raise ResolutionError(f"Unsupported mutation: {step.mutation_operation}", status="unsupported")
                
        elif step.intent == IntentAction.AGGREGATE:
            raise ResolutionError(f"Aggregations are currently unsupported.", status="unsupported")
            
        elif step.intent == IntentAction.DROP:
            if not step.columns_subject:
                raise ResolutionError("Drop intent missing columns.", status="invalid")
            cols = [self._resolve_reference(ref) for ref in step.columns_subject]
            return DropOp(columns=cols)
            
        elif step.intent == IntentAction.SELECT:
            if not step.columns_subject:
                raise ResolutionError("Select intent missing columns.", status="invalid")
            cols = [self._resolve_reference(ref) for ref in step.columns_subject]
            return SelectOp(columns=cols)
            
        else:
            raise ResolutionError(f"Unsupported intent: {step.intent}", status="unsupported")
            
    def _resolve_condition(self, node: ConditionNode) -> Condition:
        if node.node_type == NodeType.LOGICAL:
            if not node.children or not node.logic_op:
                raise ResolutionError("Logical condition missing operator or children.", status="invalid")
            child_conds = [self._resolve_condition(c) for c in node.children]
            return Condition(operator=node.logic_op.value, conditions=child_conds) # type: ignore
        
        elif node.node_type == NodeType.COMPARISON:
            if not node.subject or not node.operator:
                raise ResolutionError("Comparison missing subject or operator.", status="invalid")
            col = self._resolve_reference(node.subject, node.value)
            
            op = node.operator.value
            val = node.value.parsed_value if node.value else None
            
            if op == "temporal_match":
                raise ResolutionError("Complex temporal match is not fully supported by execution yet. Please specify explicit bounds.", status="needs_clarification")
                
            op_map = {
                "greater_than": "gt",
                "less_than": "lt",
                "greater_than_or_equals": "gte",
                "less_than_or_equals": "lte",
                "equals": "equals",
                "not_equals": "not_equals",
                "contains": "contains",
                "starts_with": "starts_with",
                "ends_with": "ends_with",
                "is_null": "is_null"
            }
            mapped_op = op_map.get(op, op)
                
            return Condition(operator=mapped_op, column=col, value=val) # type: ignore
            
        raise ResolutionError("Unknown node type.", status="invalid")
            
    def _resolve_reference(self, ref: Reference, val: Optional[Value] = None) -> str:
        # 1. EXPLICIT_COLUMN: must resolve by exact name match only.
        #    Do not fall through to evidence loop — explicit references mean the user
        #    named a specific column; any other match would be a substitution.
        if ref.reference_type == ReferenceType.EXPLICIT_COLUMN:
            target = (ref.original_text or "").lower().replace(" ", "")
            exact = [c.name for c in self.context.columns if c.name.lower().replace(" ", "") == target]
            if len(exact) == 1:
                return exact[0]
            ref_txt = ref.original_text or "Unknown"
            if len(exact) > 1:
                raise ResolutionError(f"Reference '{ref_txt}' matches multiple columns: {', '.join(exact)}.")
            raise ResolutionError(f"Could not find any column matching reference '{ref_txt}'.")

        candidates = []
        ref_text = (ref.original_text or "").lower()

        for col in self.context.columns:
            has_semantic_identity = False
            value_violates = False

            # --- STRONG EVIDENCE (establishes semantic identity) ---

            # A. Column name substring match
            col_name_lower = col.name.lower()
            if ref_text and (col_name_lower in ref_text or ref_text in col_name_lower):
                has_semantic_identity = True

            # B. Categorical value match in reference text or in explicit value
            if col.top_categories:
                if any(str(cat).lower() in ref_text for cat in col.top_categories):
                    has_semantic_identity = True
                if (val and val.semantic_type == SemanticValueType.TEXT
                        and val.parsed_value in col.top_categories):
                    has_semantic_identity = True

            # C. Semantic role keyword match
            # Uses the role string tokens (e.g. "date", "age", "monetary", "amount") as generic clues.
            # Excluded roles that carry no discriminating signal ("unknown", "categorical",
            # "numeric", "identifier") to avoid false matches.
            if col.semantic_role and col.semantic_role not in (
                "unknown", "categorical", "numeric", "identifier"
            ):
                role_parts = col.semantic_role.replace("/", " ").split()
                if any(len(p) >= 3 and p in ref_text for p in role_parts):
                    has_semantic_identity = True

            # --- SUPPORTING EVIDENCE (filters/violates, never creates a candidate) ---

            if val:
                if val.semantic_type == SemanticValueType.NUMBER:
                    if (pd.api.types.is_numeric_dtype(col.dtype)
                            and col.min_val is not None
                            and col.max_val is not None):
                        try:
                            num = float(val.parsed_value)
                            if not (col.min_val <= num <= col.max_val):
                                value_violates = True
                        except (ValueError, TypeError):
                            value_violates = True
                    else:
                        value_violates = True

                elif val.semantic_type == SemanticValueType.TEXT:
                    in_categories = (col.top_categories and val.parsed_value in col.top_categories)
                    in_df = (str(val.parsed_value) in self.df[col.name].astype(str).unique())
                    if not in_categories and not in_df:
                        value_violates = True

                elif val.semantic_type in (SemanticValueType.MONTH, SemanticValueType.DATE):
                    if col.semantic_role != "date/datetime":
                        value_violates = True

            # D. Implicit-from-value identity promotion
            # When reference_type=IMPLICIT_FROM_VALUE, original_text is null by design.
            # The value itself IS the full semantic signal. Promote value evidence to identity
            # for this reference type only, since there is no ref_text to match against.
            if not has_semantic_identity and ref.reference_type == ReferenceType.IMPLICIT_FROM_VALUE and val:
                if val.semantic_type == SemanticValueType.TEXT:
                    in_categories = (col.top_categories and val.parsed_value in col.top_categories)
                    in_df = (str(val.parsed_value) in self.df[col.name].astype(str).unique())
                    if in_categories or in_df:
                        has_semantic_identity = True
                elif val.semantic_type == SemanticValueType.NUMBER:
                    if (pd.api.types.is_numeric_dtype(col.dtype)
                            and col.min_val is not None and col.max_val is not None):
                        try:
                            num = float(val.parsed_value)
                            if col.min_val <= num <= col.max_val:
                                has_semantic_identity = True
                        except (ValueError, TypeError):
                            pass
                elif val.semantic_type in (SemanticValueType.MONTH, SemanticValueType.DATE):
                    if col.semantic_role == "date/datetime":
                        has_semantic_identity = True

            # Candidate requires BOTH semantic identity AND no value violation
            if has_semantic_identity and not value_violates:
                candidates.append(col.name)

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            ref_txt = ref.original_text or (val.original_text if val else "Unknown")
            raise ResolutionError(f"Reference '{ref_txt}' matches multiple columns: {', '.join(candidates)}.")
        else:
            ref_txt = ref.original_text or (val.original_text if val else "Unknown")
            raise ResolutionError(f"Could not find any column matching reference '{ref_txt}'.")


class NaturalLanguageInterpreter:
    def __init__(self, provider: AIProvider):
        self.provider = provider
        self.schema = FlatInterpretationIR.model_json_schema()

    def interpret(self, instruction: str, context: DatasetContext, df: pd.DataFrame) -> InterpreterResult:
        if df is None:
            raise ValueError("Staged interpretation requires the actual dataframe for deterministic resolution.")
            
        sys_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(instruction)
        
        try:
            raw_response = self.provider.generate_structured_json(sys_prompt, user_prompt, self.schema)
            parsed_json = json.loads(raw_response)
            flat_ir = FlatInterpretationIR.model_validate(parsed_json)
            ir = compile_flat_ir(flat_ir)
        except json.JSONDecodeError as e:
            return InterpreterResult(status="invalid", errors=[f"Invalid JSON returned by LLM: {e}"])
        except ValidationError as e:
            return InterpreterResult(status="invalid", errors=[f"LLM output violated schema: {e}"])
        except Exception as e:
            return InterpreterResult(status="invalid", errors=[f"Provider error: {e}"])
            
        # Stage 2: Resolution
        resolver = Stage2Resolver(context, df)
        try:
            operations = resolver.resolve(ir)
        except ResolutionError as e:
            return InterpreterResult(status=e.status, message=e.message)
        except Exception as e:
            return InterpreterResult(status="invalid", errors=[f"Resolution failure: {e}"])
            
        # Stage 3: Contract Generation
        if not operations:
            return InterpreterResult(status="invalid", errors=["No operations were resolved."])
            
        contract = ReadyContract(operations=operations)
        
        columns = [c.name for c in context.columns]
        val_errors = validate_contract(contract, columns)
        
        if val_errors:
            return InterpreterResult(
                status="invalid", 
                message="The resolution produced an invalid contract.", 
                errors=val_errors
            )
            
        return InterpreterResult(status="ready", contract=contract)
        
    def _build_system_prompt(self) -> str:
        return """You are an expert NLP parser. Your ONLY job is to extract the linguistic intent from the user's natural language instruction and map it to the provided JSON schema.
        
CRITICAL INTENT RULES:
- "Drop/remove a column" -> intent = drop
- "Drop/remove/exclude any record/row where..." -> intent = filter, filter_mode = remove
- "Keep/retain rows where..." -> intent = filter, filter_mode = keep

INTENT MAPPING RULES:
- FILTER: Must include "filter_mode" (keep/remove) and "conditions" (a list of flat conditions). If multiple conditions, specify "logic" (e.g. "and"). Do NOT nest conditions.
- SORT: Must include "sort_subject" (a reference) and "sort_direction".
- MUTATE: Must include "mutate_subject" (a reference) and "mutation_operation".
- DROP / SELECT: Must include "columns_subject" (a list of references).
- DEDUPLICATE: No subjects needed.
- AGGREGATE: Must include "aggregate_function" and "aggregate_subject" or "columns_subject".

REFERENCE RULES:
- "explicit_column": User literally mentions a specific property (e.g. "the ID column").
- "vague_concept": User describes a concept (e.g. "date", "spent", "name").
- "implicit_from_value": User just gives a value (e.g. "below 18", "Bangalore").

Extract exactly what the user asked for. Return the flat schema. Do NOT guess actual column names from the dataset. Do NOT evaluate if an operation is supported (e.g. extract aggregations strictly as "aggregate").
"""

    def _build_user_prompt(self, instruction: str) -> str:
        return f"""User Instruction:
'{instruction}'

Extract the linguistic IR JSON."""

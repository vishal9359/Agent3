"""
Stage 2: Leaf-Level Semantic Extraction (BOTTOM LEVEL)

This module extracts atomic semantic actions from the deepest AST/CFG level:
- Validation
- Permission check
- State mutation
- Irreversible side effect
- Early exit

Rules:
- NO Mermaid generation
- NO hierarchy construction yet
- NO LLM inference
- Purely deterministic, rule-based extraction
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

try:
    from clang.cindex import Cursor, CursorKind, TokenKind
except ImportError:
    raise ImportError("libclang is required. Install with: pip install libclang")

from .clang_analyzer import CFGNode, FunctionCFG, NodeType
from .call_graph_builder import FunctionInfo, CallGraph
from .cfg_builder import ControlFlowGraph, BasicBlock

logger = logging.getLogger(__name__)


class SemanticActionType(Enum):
    """Types of atomic semantic actions"""
    VALIDATION = "validation"
    PERMISSION_CHECK = "permission_check"
    STATE_MUTATION = "state_mutation"
    SIDE_EFFECT = "side_effect"  # Irreversible side effect
    EARLY_EXIT = "early_exit"
    LOOP_CONTROL = "loop_control"
    ERROR_HANDLING = "error_handling"
    DATA_ACCESS = "data_access"
    COMPUTATION = "computation"
    INITIALIZATION = "initialization"


@dataclass
class SemanticAction:
    """
    Represents a single atomic semantic action.
    This is the fundamental unit of semantic understanding.
    """
    action_type: SemanticActionType
    effect: str  # Human-readable description of what happens
    control_impact: bool  # Does this affect control flow?
    state_impact: bool  # Does this modify persistent state?
    
    # Source location
    node_id: str
    source_range: Optional[tuple] = None
    
    # Related information
    variables_read: List[str] = field(default_factory=list)
    variables_written: List[str] = field(default_factory=list)
    functions_called: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    
    # Additional metadata
    is_critical: bool = False  # Critical path operation
    is_error_path: bool = False  # Part of error handling
    confidence: float = 1.0  # Confidence in classification (0-1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "actionType": self.action_type.value,
            "effect": self.effect,
            "controlImpact": self.control_impact,
            "stateImpact": self.state_impact,
            "nodeId": self.node_id,
            "variablesRead": self.variables_read,
            "variablesWritten": self.variables_written,
            "functionsCalled": self.functions_called,
            "conditions": self.conditions,
            "isCritical": self.is_critical,
            "isErrorPath": self.is_error_path,
            "confidence": self.confidence
        }


@dataclass
class FunctionSemantics:
    """
    Semantic description of a function containing all extracted actions.
    """
    function_name: str
    is_leaf: bool = False
    actions: List[SemanticAction] = field(default_factory=list)
    exit_actions: List[SemanticAction] = field(default_factory=list)
    entry_action: Optional[SemanticAction] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "functionName": self.function_name,
            "isLeaf": self.is_leaf,
            "actions": [action.to_dict() for action in self.actions],
            "exitActions": [action.to_dict() for action in self.exit_actions],
            "entryAction": self.entry_action.to_dict() if self.entry_action else None
        }


class LeafSemanticExtractor:
    """
    Extracts atomic semantic actions from CFG nodes.
    Operates purely on AST facts, no LLM inference.
    """
    
    # Validation indicators
    VALIDATION_KEYWORDS = {
        'valid', 'check', 'verify', 'validate', 'assert',
        'ensure', 'require', 'expect', 'test'
    }
    
    # Permission/authorization indicators
    PERMISSION_KEYWORDS = {
        'permission', 'authorize', 'auth', 'allow', 'permit',
        'access', 'can', 'may', 'has_permission', 'is_allowed'
    }
    
    # State mutation indicators
    STATE_MUTATION_KEYWORDS = {
        'set', 'update', 'modify', 'change', 'alter', 'assign',
        'insert', 'delete', 'remove', 'add', 'append', 'push',
        'pop', 'clear', 'reset', 'save', 'store', 'write'
    }
    
    # Side effect indicators (irreversible)
    SIDE_EFFECT_KEYWORDS = {
        'send', 'publish', 'notify', 'broadcast', 'emit',
        'commit', 'persist', 'flush', 'sync', 'execute',
        'launch', 'start', 'stop', 'kill', 'destroy'
    }
    
    # Error handling indicators
    ERROR_KEYWORDS = {
        'error', 'fail', 'exception', 'throw', 'catch',
        'abort', 'reject', 'deny', 'refuse', 'invalid'
    }
    
    def __init__(self):
        self.semantic_actions: Dict[str, List[SemanticAction]] = {}
        logger.info("LeafSemanticExtractor initialized")
    
    def extract_from_cfg(self, cfg: FunctionCFG) -> List[SemanticAction]:
        """
        Extract all semantic actions from a function's CFG.
        
        Args:
            cfg: FunctionCFG to analyze
            
        Returns:
            List of SemanticAction objects
        """
        actions = []
        
        for node_id, node in cfg.nodes.items():
            node_actions = self._extract_from_node(node, cfg)
            actions.extend(node_actions)
        
        self.semantic_actions[cfg.qualified_name] = actions
        logger.debug(f"Extracted {len(actions)} semantic actions from {cfg.qualified_name}")
        
        return actions
    
    def _extract_from_node(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a single CFG node"""
        
        actions = []
        
        # Handle different node types
        if node.node_type == NodeType.DECISION:
            actions.extend(self._extract_from_decision(node, cfg))
        
        elif node.node_type == NodeType.CALL:
            actions.extend(self._extract_from_call(node, cfg))
        
        elif node.node_type == NodeType.RETURN:
            actions.extend(self._extract_from_return(node, cfg))
        
        elif node.node_type == NodeType.STATEMENT:
            actions.extend(self._extract_from_statement(node, cfg))
        
        elif node.node_type == NodeType.LOOP:
            actions.extend(self._extract_from_loop(node, cfg))
        
        return actions
    
    def _extract_from_decision(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a decision node"""
        
        if not node.cursor:
            return []
        
        # Get condition text
        condition_text = self._get_node_text(node)
        
        # Analyze the condition to determine semantic type
        action_type = self._classify_condition(condition_text, node)
        
        # Determine effect description
        effect = f"Check condition: {condition_text}"
        if action_type == SemanticActionType.VALIDATION:
            effect = f"Validate: {condition_text}"
        elif action_type == SemanticActionType.PERMISSION_CHECK:
            effect = f"Check permission: {condition_text}"
        elif action_type == SemanticActionType.ERROR_HANDLING:
            effect = f"Check for error: {condition_text}"
        
        # Extract variables referenced in condition
        variables = self._extract_variables(node.cursor)
        
        action = SemanticAction(
            action_type=action_type,
            effect=effect,
            control_impact=True,  # Decisions always affect control flow
            state_impact=False,
            node_id=node.id,
            source_range=node.source_range,
            variables_read=variables,
            conditions=[condition_text],
            is_critical=self._is_critical_decision(node, cfg)
        )
        
        return [action]
    
    def _extract_from_call(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a function call node"""
        
        if not node.cursor or not node.called_functions:
            return []
        
        actions = []
        
        for func_name in node.called_functions:
            # Classify the call based on function name
            action_type = self._classify_function_call(func_name)
            
            # Determine effect
            effect = f"Call {func_name}"
            if action_type == SemanticActionType.VALIDATION:
                effect = f"Validate using {func_name}"
            elif action_type == SemanticActionType.STATE_MUTATION:
                effect = f"Update state via {func_name}"
            elif action_type == SemanticActionType.SIDE_EFFECT:
                effect = f"Execute side effect: {func_name}"
            elif action_type == SemanticActionType.DATA_ACCESS:
                effect = f"Access data via {func_name}"
            
            # Extract arguments
            args = self._extract_call_arguments(node.cursor)
            
            action = SemanticAction(
                action_type=action_type,
                effect=effect,
                control_impact=action_type in [
                    SemanticActionType.VALIDATION,
                    SemanticActionType.EARLY_EXIT,
                    SemanticActionType.ERROR_HANDLING
                ],
                state_impact=action_type in [
                    SemanticActionType.STATE_MUTATION,
                    SemanticActionType.SIDE_EFFECT
                ],
                node_id=node.id,
                source_range=node.source_range,
                functions_called=[func_name],
                variables_read=args,
                is_critical=self._is_critical_call(func_name)
            )
            
            actions.append(action)
        
        return actions
    
    def _extract_from_return(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a return statement"""
        
        # Check if this is an early exit (not at end of function)
        is_early = len(node.predecessors) > 0 and not self._is_at_function_end(node, cfg)
        
        # Check if this is an error return
        is_error = self._is_error_return(node)
        
        action_type = SemanticActionType.EARLY_EXIT if is_early else SemanticActionType.COMPUTATION
        if is_error:
            action_type = SemanticActionType.ERROR_HANDLING
        
        return_value = self._get_return_value(node)
        effect = f"Return {return_value}" if return_value else "Return"
        
        if is_error:
            effect = f"Return error: {return_value}"
        elif is_early:
            effect = f"Early exit with {return_value}"
        
        action = SemanticAction(
            action_type=action_type,
            effect=effect,
            control_impact=True,
            state_impact=False,
            node_id=node.id,
            source_range=node.source_range,
            is_error_path=is_error,
            is_critical=is_early or is_error
        )
        
        return [action]
    
    def _extract_from_statement(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a general statement"""
        
        if not node.cursor:
            return []
        
        # Check if this is a state mutation
        if node.mutates_state:
            variables = self._extract_variables(node.cursor)
            written_vars = self._extract_written_variables(node.cursor)
            
            action = SemanticAction(
                action_type=SemanticActionType.STATE_MUTATION,
                effect=f"Modify state: {', '.join(written_vars) if written_vars else 'variables'}",
                control_impact=False,
                state_impact=True,
                node_id=node.id,
                source_range=node.source_range,
                variables_read=variables,
                variables_written=written_vars
            )
            
            return [action]
        
        # Check for initialization
        if self._is_initialization(node):
            action = SemanticAction(
                action_type=SemanticActionType.INITIALIZATION,
                effect=f"Initialize: {self._get_node_text(node)}",
                control_impact=False,
                state_impact=True,
                node_id=node.id,
                source_range=node.source_range
            )
            return [action]
        
        # Default: computation
        action = SemanticAction(
            action_type=SemanticActionType.COMPUTATION,
            effect=f"Compute: {self._get_node_text(node)}",
            control_impact=False,
            state_impact=False,
            node_id=node.id,
            source_range=node.source_range
        )
        
        return [action]
    
    def _extract_from_loop(self, node: CFGNode, cfg: FunctionCFG) -> List[SemanticAction]:
        """Extract semantic actions from a loop node"""
        
        condition_text = self._get_node_text(node)
        
        action = SemanticAction(
            action_type=SemanticActionType.LOOP_CONTROL,
            effect=f"Loop while: {condition_text}",
            control_impact=True,
            state_impact=False,
            node_id=node.id,
            source_range=node.source_range,
            conditions=[condition_text]
        )
        
        return [action]
    
    # Classification helpers
    
    def _classify_condition(self, condition_text: str, node: CFGNode) -> SemanticActionType:
        """Classify a conditional expression"""
        
        text_lower = condition_text.lower()
        
        # Check for validation patterns
        if any(kw in text_lower for kw in self.VALIDATION_KEYWORDS):
            return SemanticActionType.VALIDATION
        
        # Check for permission patterns
        if any(kw in text_lower for kw in self.PERMISSION_KEYWORDS):
            return SemanticActionType.PERMISSION_CHECK
        
        # Check for error patterns
        if any(kw in text_lower for kw in self.ERROR_KEYWORDS):
            return SemanticActionType.ERROR_HANDLING
        
        # Check for null/empty checks (common validation)
        if any(pattern in text_lower for pattern in ['null', 'nullptr', 'empty', '== 0', '!=']):
            return SemanticActionType.VALIDATION
        
        # Default: general validation
        return SemanticActionType.VALIDATION
    
    def _classify_function_call(self, func_name: str) -> SemanticActionType:
        """Classify a function call based on its name"""
        
        name_lower = func_name.lower()
        
        # Check each category
        if any(kw in name_lower for kw in self.VALIDATION_KEYWORDS):
            return SemanticActionType.VALIDATION
        
        if any(kw in name_lower for kw in self.PERMISSION_KEYWORDS):
            return SemanticActionType.PERMISSION_CHECK
        
        if any(kw in name_lower for kw in self.STATE_MUTATION_KEYWORDS):
            return SemanticActionType.STATE_MUTATION
        
        if any(kw in name_lower for kw in self.SIDE_EFFECT_KEYWORDS):
            return SemanticActionType.SIDE_EFFECT
        
        if any(kw in name_lower for kw in self.ERROR_KEYWORDS):
            return SemanticActionType.ERROR_HANDLING
        
        # Check for getter patterns
        if name_lower.startswith('get') or name_lower.startswith('find') or name_lower.startswith('lookup'):
            return SemanticActionType.DATA_ACCESS
        
        # Default: computation
        return SemanticActionType.COMPUTATION
    
    # Helper methods
    
    def _get_node_text(self, node: CFGNode) -> str:
        """Extract text representation of a node"""
        if node.text:
            return node.text
        
        if node.cursor:
            try:
                tokens = list(node.cursor.get_tokens())
                return ' '.join(t.spelling for t in tokens[:20])  # Limit to first 20 tokens
            except:
                return node.cursor.spelling or "<?>"
        
        return "<?>"
    
    def _extract_variables(self, cursor: Cursor) -> List[str]:
        """Extract variable names referenced in a cursor"""
        variables = []
        
        def visit(c: Cursor):
            if c.kind in [CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR]:
                if c.spelling:
                    variables.append(c.spelling)
            
            for child in c.get_children():
                visit(child)
        
        visit(cursor)
        return list(set(variables))  # Remove duplicates
    
    def _extract_written_variables(self, cursor: Cursor) -> List[str]:
        """Extract variables being written to"""
        written = []
        
        # Look for left-hand side of assignments
        if cursor.kind in [CursorKind.BINARY_OPERATOR, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR]:
            children = list(cursor.get_children())
            if children:
                lhs = children[0]
                if lhs.kind in [CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR]:
                    if lhs.spelling:
                        written.append(lhs.spelling)
        
        return written
    
    def _extract_call_arguments(self, cursor: Cursor) -> List[str]:
        """Extract argument expressions from a call"""
        args = []
        
        for child in cursor.get_children():
            if child.kind not in [CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR]:
                # This is an argument
                vars_in_arg = self._extract_variables(child)
                args.extend(vars_in_arg)
        
        return args
    
    def _get_return_value(self, node: CFGNode) -> str:
        """Extract return value from a return statement"""
        if not node.cursor:
            return ""
        
        children = list(node.cursor.get_children())
        if children:
            return self._get_node_text(CFGNode(
                id="temp",
                node_type=NodeType.STATEMENT,
                cursor=children[0]
            ))
        
        return ""
    
    def _is_error_return(self, node: CFGNode) -> bool:
        """Check if a return statement indicates an error"""
        return_val = self._get_return_value(node).lower()
        
        # Common error indicators
        error_indicators = [
            'false', 'null', 'nullptr', '-1', 'error',
            'fail', 'status::error', 'absl::status'
        ]
        
        return any(indicator in return_val for indicator in error_indicators)
    
    def _is_at_function_end(self, node: CFGNode, cfg: FunctionCFG) -> bool:
        """Check if a node is at the end of the function"""
        # Simple heuristic: check if there are no successors or only exit nodes
        return len(node.successors) == 0
    
    def _is_critical_decision(self, node: CFGNode, cfg: FunctionCFG) -> bool:
        """Determine if a decision is critical to the function's logic"""
        # Heuristic: decisions that lead to early returns or error handling
        for succ_id in node.successors:
            succ = cfg.nodes.get(succ_id)
            if succ and succ.node_type == NodeType.RETURN:
                return True
        
        return False
    
    def _is_critical_call(self, func_name: str) -> bool:
        """Determine if a function call is critical"""
        name_lower = func_name.lower()
        
        critical_indicators = [
            'validate', 'check', 'verify', 'authorize',
            'commit', 'persist', 'send', 'publish',
            'execute', 'create', 'delete', 'update'
        ]
        
        return any(indicator in name_lower for indicator in critical_indicators)
    
    def _is_initialization(self, node: CFGNode) -> bool:
        """Check if a statement is an initialization"""
        if not node.cursor:
            return False
        
        return node.cursor.kind in [
            CursorKind.VAR_DECL,
            CursorKind.PARM_DECL,
            CursorKind.FIELD_DECL
        ]
    
    def get_actions_for_function(self, qualified_name: str) -> List[SemanticAction]:
        """Retrieve semantic actions for a specific function"""
        return self.semantic_actions.get(qualified_name, [])
    
    def get_critical_actions(self, qualified_name: str) -> List[SemanticAction]:
        """Get only critical semantic actions for a function"""
        actions = self.get_actions_for_function(qualified_name)
        return [a for a in actions if a.is_critical or a.control_impact or a.state_impact]
    """
    Extracts atomic semantic actions from CFG blocks.
    This is Stage 2: Leaf-Level Semantic Extraction.
    """
    
    def __init__(self):
        pass
    
    def extract_function_semantics(self, func_info: FunctionInfo, cfg: ControlFlowGraph) -> FunctionSemantics:
        """
        Extract semantic actions from a function's CFG.
        
        Args:
            func_info: Information about the function
            cfg: Control Flow Graph of the function
            
        Returns:
            FunctionSemantics containing all atomic actions
        """
        logger.debug(f"Extracting semantics for function: {func_info.name}")
        
        semantics = FunctionSemantics(
            function_name=func_info.qualified_name,
            is_leaf=func_info.is_leaf
        )
        
        # Process each basic block
        for block_id, block in cfg.blocks.items():
            block_actions = self._extract_block_semantics(block, cfg)
            semantics.actions.extend(block_actions)
            
            # Identify exit actions
            if block_id in cfg.exit_block_ids:
                semantics.exit_actions.extend(block_actions)
        
        # Identify entry action (first block's first action)
        if cfg.entry_block_id in cfg.blocks:
            entry_block = cfg.blocks[cfg.entry_block_id]
            entry_actions = self._extract_block_semantics(entry_block, cfg)
            if entry_actions:
                semantics.entry_action = entry_actions[0]
        
        logger.debug(f"Extracted {len(semantics.actions)} semantic actions from {func_info.name}")
        
        return semantics
    
    def _extract_block_semantics(self, block: BasicBlock, cfg: ControlFlowGraph) -> List[SemanticAction]:
        """Extract semantic actions from a basic block"""
        actions = []
        
        # Process each statement in the block
        for stmt in block.statements:
            action = self._extract_statement_semantics(stmt, block)
            if action:
                actions.append(action)
        
        # If block is a guard, create a decision action
        if block.is_guard and block.guard_condition:
            decision_action = SemanticAction(
                action_type=SemanticActionType.DECISION,
                effect=f"Decision point: {block.guard_condition}",
                control_impact=True,
                state_impact=False,
                source_location=block.statements[0].location if block.statements else "unknown",
                guard_condition=block.guard_condition,
                metadata={
                    "block_id": block.id,
                    "has_validation": block.has_validation,
                    "successors": list(block.successors),
                    "edge_types": {str(k): v.value for k, v in block.edge_types.items()}
                }
            )
            actions.append(decision_action)
        
        # If block is error exit, create early exit action
        if block.is_error_exit:
            exit_action = SemanticAction(
                action_type=SemanticActionType.EARLY_EXIT,
                effect="Error exit: terminate execution with error",
                control_impact=True,
                state_impact=False,
                source_location=block.statements[-1].location if block.statements else "unknown",
                metadata={"block_id": block.id, "is_error": True}
            )
            actions.append(exit_action)
        
        return actions
    
    def _extract_statement_semantics(self, stmt: ASTNode, block: BasicBlock) -> Optional[SemanticAction]:
        """Extract semantic action from a single statement"""
        node_type = stmt.node_type
        spelling = stmt.spelling
        location = stmt.location
        
        # Validation
        if node_type == NodeType.VALIDATION:
            return SemanticAction(
                action_type=SemanticActionType.VALIDATION,
                effect=f"Validate: {spelling}",
                control_impact=True,  # Validation typically affects control flow
                state_impact=False,
                source_location=location,
                code_snippet=spelling,
                metadata={"validation_function": spelling}
            )
        
        # Permission check
        elif node_type == NodeType.PERMISSION_CHECK:
            return SemanticAction(
                action_type=SemanticActionType.PERMISSION_CHECK,
                effect=f"Check permission: {spelling}",
                control_impact=True,
                state_impact=False,
                source_location=location,
                code_snippet=spelling,
                metadata={"permission_function": spelling}
            )
        
        # State mutation
        elif node_type == NodeType.STATE_MUTATION:
            return SemanticAction(
                action_type=SemanticActionType.STATE_MUTATION,
                effect=f"Modify state: {spelling}",
                control_impact=False,
                state_impact=True,
                source_location=location,
                code_snippet=spelling,
                metadata={"mutation_type": "state_change"}
            )
        
        # Irreversible side effect
        elif node_type == NodeType.IRREVERSIBLE_SIDE_EFFECT:
            return SemanticAction(
                action_type=SemanticActionType.IRREVERSIBLE_SIDE_EFFECT,
                effect=f"Irreversible operation: {spelling}",
                control_impact=False,
                state_impact=True,
                source_location=location,
                code_snippet=spelling,
                metadata={"side_effect_function": spelling}
            )
        
        # Early exit
        elif node_type == NodeType.EARLY_EXIT:
            return SemanticAction(
                action_type=SemanticActionType.EARLY_EXIT,
                effect=f"Early exit: {spelling}",
                control_impact=True,
                state_impact=False,
                source_location=location,
                code_snippet=spelling
            )
        
        # Function call
        elif node_type == NodeType.FUNCTION_CALL:
            return SemanticAction(
                action_type=SemanticActionType.FUNCTION_CALL,
                effect=f"Call function: {spelling}",
                control_impact=False,  # May be updated during aggregation
                state_impact=False,  # May be updated during aggregation
                source_location=location,
                code_snippet=spelling,
                metadata={"called_function": spelling}
            )
        
        # Unknown - skip for now
        return None


def extract_project_semantics(call_graph: CallGraph, cfgs: Dict[str, ControlFlowGraph]) -> Dict[str, FunctionSemantics]:
    """
    Extract semantic actions for all functions in a project.
    
    Args:
        call_graph: CallGraph from ClangASTParser
        cfgs: Dictionary of CFGs from CFGBuilder
        
    Returns:
        Dictionary mapping function names to their FunctionSemantics
    """
    extractor = SemanticExtractor()
    semantics_map = {}
    
    logger.info(f"Extracting semantics for {len(call_graph.functions)} functions")
    
    for func_name, func_info in call_graph.functions.items():
        if func_name in cfgs:
            try:
                semantics = extractor.extract_function_semantics(func_info, cfgs[func_name])
                semantics_map[func_name] = semantics
            except Exception as e:
                logger.error(f"Failed to extract semantics for {func_name}: {e}")
    
    logger.info(f"Successfully extracted semantics for {len(semantics_map)} functions")
    
    # Log statistics
    leaf_count = sum(1 for s in semantics_map.values() if s.is_leaf)
    total_actions = sum(len(s.actions) for s in semantics_map.values())
    logger.info(f"Statistics: {leaf_count} leaf functions, {total_actions} total semantic actions")
    
    return semantics_map

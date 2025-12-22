"""
Leaf-Level Semantic Extractor - Stage 2

This module extracts atomic semantic actions from the deepest AST/CFG level.
It identifies:
- validation actions
- permission checks
- state mutations
- irreversible side effects
- early exits

Each atomic unit produces a local semantic description object.

NO Mermaid, NO hierarchy yet, NO LLM inference at this stage.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

try:
    # Try importing from clang_ast_extractor first (used by docagent_pipeline)
    from .clang_ast_extractor import BasicBlock, ControlFlowGraph as FunctionCFG, ProjectAST
except ImportError:
    # Fallback to clang_ast_parser
    from .clang_ast_parser import BasicBlock, FunctionCFG, ProjectAST

logger = logging.getLogger(__name__)


class SemanticActionType(Enum):
    """Types of atomic semantic actions."""
    VALIDATION = "validation"
    PERMISSION_CHECK = "permission_check"
    STATE_MUTATION = "state_mutation"
    IRREVERSIBLE_SIDE_EFFECT = "irreversible_side_effect"
    EARLY_EXIT = "early_exit"
    DATA_RETRIEVAL = "data_retrieval"
    COMPUTATION = "computation"
    LOGGING = "logging"  # Noise
    METRIC = "metric"  # Noise
    UTILITY = "utility"  # Noise


@dataclass
class SemanticAction:
    """
    Represents a single atomic semantic action.
    
    This is the fundamental unit of semantic understanding.
    """
    action_type: SemanticActionType
    effect: str  # Human-readable description of what this action does
    control_impact: bool  # Does this affect control flow?
    state_impact: bool  # Does this modify persistent state?
    source_statement: str = ""
    function_name: str = ""
    block_id: str = ""
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class BlockSemantics:
    """Semantic description of a basic block."""
    block_id: str
    function_name: str
    actions: List[SemanticAction] = field(default_factory=list)
    is_leaf: bool = False
    has_condition: bool = False
    condition_semantics: Optional[str] = None


@dataclass
class FunctionSemantics:
    """Semantic description of a function (leaf-level)."""
    function_name: str
    file_path: str
    is_leaf: bool = False
    blocks: Dict[str, BlockSemantics] = field(default_factory=dict)
    dominant_action_type: Optional[SemanticActionType] = None
    summary: str = ""  # Will be filled by bottom-up aggregation


class LeafSemanticExtractor:
    """
    Extracts atomic semantic actions from AST/CFG at the deepest level.
    
    This is a rule-based, deterministic extractor. No LLM is used here.
    """

    def __init__(self, project_ast: ProjectAST):
        """
        Initialize the extractor.
        
        Args:
            project_ast: Complete AST from ClangASTParser
        """
        self.project_ast = project_ast
        self.function_semantics: Dict[str, FunctionSemantics] = {}
        
        # Semantic patterns (rule-based)
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for semantic classification."""
        # Validation patterns
        self.validation_patterns = [
            (re.compile(r"\bcheck\w*\(", re.I), "validation check"),
            (re.compile(r"\bvalidate\w*\(", re.I), "validation"),
            (re.compile(r"\bisvalid\w*\(", re.I), "validity check"),
            (re.compile(r"\bverify\w*\(", re.I), "verification"),
            (re.compile(r"\bparse\w*\(", re.I), "parsing/validation"),
            (re.compile(r"if\s*\(\s*!\s*\w+", re.I), "negative condition check"),
            (re.compile(r"if\s*\(\s*\w+\s*==\s*null", re.I), "null check"),
            (re.compile(r"if\s*\(\s*\w+\s*<\s*0", re.I), "error value check"),
        ]
        
        # Permission check patterns
        self.permission_patterns = [
            (re.compile(r"\bhas\w*permission\w*\(", re.I), "permission check"),
            (re.compile(r"\bcan\w*\(", re.I), "capability check"),
            (re.compile(r"\ballow\w*\(", re.I), "access control"),
            (re.compile(r"\bauthorize\w*\(", re.I), "authorization"),
        ]
        
        # State mutation patterns
        self.state_mutation_patterns = [
            (re.compile(r"\bset\w*\(", re.I), "set state"),
            (re.compile(r"\bupdate\w*\(", re.I), "update state"),
            (re.compile(r"\bmodify\w*\(", re.I), "modify state"),
            (re.compile(r"\binsert\w*\(", re.I), "insert"),
            (re.compile(r"\badd\w*\(", re.I), "add"),
            (re.compile(r"\bremove\w*\(", re.I), "remove"),
            (re.compile(r"\bdelete\w*\(", re.I), "delete"),
            (re.compile(r"\berase\w*\(", re.I), "erase"),
            (re.compile(r"\bcreate\w*\(", re.I), "create"),
            (re.compile(r"\bdestroy\w*\(", re.I), "destroy"),
            (re.compile(r"\w+\s*=\s*[^=]", re.I), "assignment"),
        ]
        
        # Irreversible side effect patterns
        self.irreversible_patterns = [
            (re.compile(r"\bwrite\w*\(", re.I), "write operation"),
            (re.compile(r"\bcommit\w*\(", re.I), "commit"),
            (re.compile(r"\bpersist\w*\(", re.I), "persist"),
            (re.compile(r"\bsend\w*\(", re.I), "send message"),
            (re.compile(r"\bpublish\w*\(", re.I), "publish"),
            (re.compile(r"\bexecute\w*\(", re.I), "execute action"),
        ]
        
        # Early exit patterns
        self.early_exit_patterns = [
            (re.compile(r"\breturn\b", re.I), "return"),
            (re.compile(r"\bthrow\b", re.I), "exception throw"),
            (re.compile(r"\bexit\w*\(", re.I), "exit"),
            (re.compile(r"\babort\w*\(", re.I), "abort"),
        ]
        
        # Data retrieval patterns
        self.data_retrieval_patterns = [
            (re.compile(r"\bget\w*\(", re.I), "get data"),
            (re.compile(r"\bfetch\w*\(", re.I), "fetch"),
            (re.compile(r"\bretrieve\w*\(", re.I), "retrieve"),
            (re.compile(r"\bread\w*\(", re.I), "read"),
            (re.compile(r"\bload\w*\(", re.I), "load"),
            (re.compile(r"\bquery\w*\(", re.I), "query"),
        ]
        
        # Noise patterns (to filter out)
        self.logging_patterns = [
            (re.compile(r"\blog\w*\(", re.I), "logging"),
            (re.compile(r"\bprint\w*\(", re.I), "print"),
            (re.compile(r"\bcout\s*<<", re.I), "console output"),
            (re.compile(r"\bcerr\s*<<", re.I), "error output"),
        ]
        
        self.metric_patterns = [
            (re.compile(r"\bmetric\w*\(", re.I), "metric"),
            (re.compile(r"\brecord\w*\(", re.I), "record metric"),
            (re.compile(r"\btrack\w*\(", re.I), "tracking"),
        ]
        
        self.utility_patterns = [
            (re.compile(r"\bmalloc\w*\(", re.I), "memory allocation"),
            (re.compile(r"\bfree\w*\(", re.I), "memory free"),
            (re.compile(r"\bnew\s+\w+", re.I), "object allocation"),
            (re.compile(r"\bdelete\s+\w+", re.I), "object deletion"),
            (re.compile(r"\bserialize\w*\(", re.I), "serialization"),
            (re.compile(r"\bdeserialize\w*\(", re.I), "deserialization"),
            (re.compile(r"\bto_string\w*\(", re.I), "string conversion"),
        ]
    
    def extract_semantics(self) -> Dict[str, FunctionSemantics]:
        """
        Extract semantic actions for all functions in the project.
        
        Returns:
            Dictionary mapping function names to their semantic descriptions
        """
        logger.info("Extracting leaf-level semantics from AST")
        
        # Process all functions
        for func_name, cfg in self.project_ast.function_cfgs.items():
            func_sem = self._extract_function_semantics(cfg)
            self.function_semantics[func_name] = func_sem
        
        logger.info(f"Extracted semantics for {len(self.function_semantics)} functions")
        
        # Log statistics
        leaf_count = sum(1 for fs in self.function_semantics.values() if fs.is_leaf)
        logger.info(f"  - {leaf_count} leaf functions")
        logger.info(f"  - {len(self.function_semantics) - leaf_count} non-leaf functions")
        
        return self.function_semantics
    
    def _extract_function_semantics(self, cfg: FunctionCFG) -> FunctionSemantics:
        """Extract semantic actions for a single function."""
        func_sem = FunctionSemantics(
            function_name=cfg.function_name,
            file_path=cfg.file_path,
            is_leaf=cfg.is_leaf_function,
        )
        
        # Process each basic block
        for block_id, block in cfg.blocks.items():
            block_sem = self._extract_block_semantics(block, cfg.function_name)
            func_sem.blocks[block_id] = block_sem
        
        # Determine dominant action type
        func_sem.dominant_action_type = self._determine_dominant_action(func_sem)
        
        return func_sem
    
    def _extract_block_semantics(
        self, block: BasicBlock, function_name: str
    ) -> BlockSemantics:
        """Extract semantic actions from a basic block."""
        block_sem = BlockSemantics(
            block_id=block.id,
            function_name=function_name,
            is_leaf=block.is_leaf,
            has_condition=block.has_condition,
            condition_semantics=self._classify_condition(block.condition_expr)
            if block.condition_expr
            else None,
        )
        
        # Process each statement
        # Handle both Cursor objects (from clang_ast_extractor) and strings (from clang_ast_parser)
        for stmt in block.statements:
            # Convert Cursor to string if needed
            if hasattr(stmt, 'spelling') or hasattr(stmt, 'get_tokens'):
                # It's a Cursor object, extract source text
                stmt_str = self._cursor_to_string(stmt)
            else:
                # It's already a string
                stmt_str = str(stmt)
            
            action = self._classify_statement(stmt_str, function_name, block.id)
            if action:
                block_sem.actions.append(action)
        
        return block_sem
    
    def _classify_statement(
        self, stmt: str, function_name: str, block_id: str
    ) -> Optional[SemanticAction]:
        """
        Classify a statement into a semantic action.
        
        This is the core classification logic - purely rule-based.
        """
        stmt = stmt.strip()
        
        if not stmt or stmt.startswith("//") or stmt.startswith("/*"):
            return None
        
        # Check noise patterns first (filter out)
        if self._matches_patterns(stmt, self.logging_patterns):
            return SemanticAction(
                action_type=SemanticActionType.LOGGING,
                effect="logging",
                control_impact=False,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.95,
            )
        
        if self._matches_patterns(stmt, self.metric_patterns):
            return SemanticAction(
                action_type=SemanticActionType.METRIC,
                effect="metric recording",
                control_impact=False,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.95,
            )
        
        if self._matches_patterns(stmt, self.utility_patterns):
            return SemanticAction(
                action_type=SemanticActionType.UTILITY,
                effect="utility operation",
                control_impact=False,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.9,
            )
        
        # Check early exit patterns
        if self._matches_patterns(stmt, self.early_exit_patterns):
            return SemanticAction(
                action_type=SemanticActionType.EARLY_EXIT,
                effect=f"exit function: {stmt[:50]}",
                control_impact=True,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=1.0,
            )
        
        # Check validation patterns
        if self._matches_patterns(stmt, self.validation_patterns):
            return SemanticAction(
                action_type=SemanticActionType.VALIDATION,
                effect=f"validate: {self._extract_subject(stmt)}",
                control_impact=True,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.9,
            )
        
        # Check permission patterns
        if self._matches_patterns(stmt, self.permission_patterns):
            return SemanticAction(
                action_type=SemanticActionType.PERMISSION_CHECK,
                effect=f"check permission: {self._extract_subject(stmt)}",
                control_impact=True,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.9,
            )
        
        # Check irreversible patterns
        if self._matches_patterns(stmt, self.irreversible_patterns):
            return SemanticAction(
                action_type=SemanticActionType.IRREVERSIBLE_SIDE_EFFECT,
                effect=f"irreversible action: {self._extract_subject(stmt)}",
                control_impact=False,
                state_impact=True,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.85,
            )
        
        # Check state mutation patterns
        if self._matches_patterns(stmt, self.state_mutation_patterns):
            return SemanticAction(
                action_type=SemanticActionType.STATE_MUTATION,
                effect=f"modify state: {self._extract_subject(stmt)}",
                control_impact=False,
                state_impact=True,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.85,
            )
        
        # Check data retrieval patterns
        if self._matches_patterns(stmt, self.data_retrieval_patterns):
            return SemanticAction(
                action_type=SemanticActionType.DATA_RETRIEVAL,
                effect=f"retrieve: {self._extract_subject(stmt)}",
                control_impact=False,
                state_impact=False,
                source_statement=stmt,
                function_name=function_name,
                block_id=block_id,
                confidence=0.8,
            )
        
        # Default: computation
        return SemanticAction(
            action_type=SemanticActionType.COMPUTATION,
            effect=f"compute: {stmt[:50]}",
            control_impact=False,
            state_impact=False,
            source_statement=stmt,
            function_name=function_name,
            block_id=block_id,
            confidence=0.5,
        )
    
    def _matches_patterns(self, stmt: str, patterns: List[tuple]) -> bool:
        """Check if statement matches any pattern."""
        for pattern, _ in patterns:
            if pattern.search(stmt):
                return True
        return False
    
    def _cursor_to_string(self, cursor) -> str:
        """Convert a Cursor object to source code string."""
        try:
            # Try to get tokens and join them
            tokens = list(cursor.get_tokens())
            if tokens:
                return " ".join(t.spelling for t in tokens)
        except:
            pass
        
        try:
            # Fallback to spelling
            if hasattr(cursor, 'spelling') and cursor.spelling:
                return cursor.spelling
        except:
            pass
        
        # Last resort: string representation
        return str(cursor)
    
    def _extract_subject(self, stmt: str) -> str:
        """Extract the subject/object of an action (simplified)."""
        # Try to extract function name or variable name
        match = re.search(r"(\w+)\s*\(", stmt)
        if match:
            return match.group(1)
        
        match = re.search(r"(\w+)\s*[=<>!]", stmt)
        if match:
            return match.group(1)
        
        return stmt[:30]
    
    def _classify_condition(self, condition: Optional[str]) -> Optional[str]:
        """Classify a condition expression semantically."""
        if not condition:
            return None
        
        # Validation conditions
        if any(p[0].search(condition) for p in self.validation_patterns):
            return "validation decision"
        
        # Permission conditions
        if any(p[0].search(condition) for p in self.permission_patterns):
            return "permission decision"
        
        # Null checks
        if "null" in condition.lower() or "nullptr" in condition.lower():
            return "null check"
        
        # Error checks
        if "<" in condition and "0" in condition:
            return "error check"
        
        return "condition check"
    
    def _determine_dominant_action(
        self, func_sem: FunctionSemantics
    ) -> Optional[SemanticActionType]:
        """Determine the dominant semantic action type for a function."""
        action_counts: Dict[SemanticActionType, int] = {}
        
        for block_sem in func_sem.blocks.values():
            for action in block_sem.actions:
                action_counts[action.action_type] = (
                    action_counts.get(action.action_type, 0) + 1
                )
        
        # Filter out noise
        noise_types = {
            SemanticActionType.LOGGING,
            SemanticActionType.METRIC,
            SemanticActionType.UTILITY,
        }
        significant_counts = {
            k: v for k, v in action_counts.items() if k not in noise_types
        }
        
        if not significant_counts:
            return None
        
        return max(significant_counts, key=significant_counts.get)
    
    def get_leaf_functions(self) -> List[str]:
        """Get all leaf function names."""
        return [
            name
            for name, sem in self.function_semantics.items()
            if sem.is_leaf
        ]
    
    def get_function_semantics(self, function_name: str) -> Optional[FunctionSemantics]:
        """Get semantic description for a specific function."""
        return self.function_semantics.get(function_name)


def extract_leaf_semantics(project_ast: ProjectAST) -> Dict[str, Dict[str, 'BlockSemantics']]:
    """
    Convenience function to extract leaf-level semantics from a project AST.
    
    Args:
        project_ast: Complete project AST with CFGs
    
    Returns:
        Dictionary mapping function_name -> (block_id -> BlockSemantics)
    """
    extractor = LeafSemanticExtractor(project_ast)
    function_semantics = extractor.extract_semantics()
    
    # Convert FunctionSemantics to dict of BlockSemantics for compatibility
    result = {}
    for func_name, func_sem in function_semantics.items():
        result[func_name] = func_sem.blocks
    
    return result

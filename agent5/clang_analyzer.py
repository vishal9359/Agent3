"""
Stage 1: Full AST Construction using Clang (NO LLM)

This module provides:
- Complete AST parsing for all C++ translation units
- Control Flow Graph (CFG) construction per function
- Call graph extraction
- Identification of:
  - Leaf-level execution units (basic blocks)
  - Guard conditions
  - State mutations
  - Error exits
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import clang.cindex as clang
    from clang.cindex import CursorKind, TypeKind, TokenKind
except ImportError:
    raise ImportError(
        "libclang is required for AST analysis. "
        "Install it with: pip install libclang"
    )

logger = logging.getLogger(__name__)

# Import centralized project exclusion configuration
from agent5.fs_utils import is_in_project_scope, PROJECT_EXCLUDE_DIRS, PROJECT_EXCLUDE_PATTERNS


class NodeType(Enum):
    """Types of CFG nodes"""
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    DECISION = "decision"
    LOOP = "loop"
    CALL = "call"
    RETURN = "return"
    ERROR_EXIT = "error_exit"
    EARLY_EXIT = "early_exit"
    VALIDATION = "validation"
    PERMISSION_CHECK = "permission_check"
    STATE_MUTATION = "state_mutation"
    IRREVERSIBLE_SIDE_EFFECT = "irreversible_side_effect"
    FUNCTION_CALL = "function_call"


@dataclass
class CFGNode:
    """Represents a node in the Control Flow Graph"""
    id: str
    node_type: NodeType
    cursor: Optional[clang.Cursor] = None
    source_range: Optional[Tuple[int, int]] = None
    text: str = ""
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    
    # Semantic metadata
    is_guard: bool = False
    mutates_state: bool = False
    is_error_path: bool = False
    called_functions: List[str] = field(default_factory=list)


@dataclass
class FunctionCFG:
    """Control Flow Graph for a single function"""
    function_name: str
    qualified_name: str
    file_path: str
    entry_node: str
    exit_nodes: List[str]
    nodes: Dict[str, CFGNode]
    parameters: List[Tuple[str, str]]  # [(name, type)]
    return_type: str
    is_leaf: bool = False  # True if doesn't call other functions


@dataclass
class CallRelation:
    """Represents a function call relationship"""
    caller: str  # Qualified function name
    callee: str  # Qualified function name
    call_site: str  # Location in source
    is_conditional: bool = False  # Called within if/switch/loop


# Use centralized exclusion configuration from fs_utils
# EXCLUDED_DIR_NAMES is now imported from agent5.fs_utils


class ClangAnalyzer:
    """Main analyzer for C++ projects using Clang"""
    
    def __init__(self, project_path: str, compile_commands: Optional[str] = None):
        """
        Initialize the Clang analyzer
        
        Args:
            project_path: Root path of the C++ project
            compile_commands: Path to compile_commands.json (optional)
        """
        self.project_path = Path(project_path).resolve()
        if not self.project_path.is_dir():
            raise ValueError(f"Invalid project root (not a directory): {self.project_path}")
        self.compile_commands = compile_commands
        
        # Initialize Clang index
        self.index = clang.Index.create()
        
        # Storage for analysis results
        self.translation_units: Dict[str, clang.TranslationUnit] = {}
        self.function_cfgs: Dict[str, FunctionCFG] = {}
        self.call_graph: List[CallRelation] = []
        
        # Mapping from cursor to qualified name
        self.cursor_to_name: Dict[int, str] = {}
        
        logger.info(f"ClangAnalyzer initialized for project: {project_path}")
    
    def _is_in_project_scope(self, path: Path) -> bool:
        """
        Check if a path is within the project root and not in excluded directories.
        """
        try:
            path = path.resolve()
        except Exception:
            return False

        # Use centralized exclusion check
        return is_in_project_scope(path, self.project_path)

    def analyze_project(self, file_patterns: Optional[List[str]] = None) -> None:
        """
        Analyze the entire C++ project
        
        Args:
            file_patterns: Optional list of glob patterns (e.g., ['*.cpp', '*.cc'])
        """
        if file_patterns is None:
            file_patterns = ['*.cpp', '*.cc', '*.cxx', '*.c++']
        
        # Find all C++ source files strictly within project root
        cpp_files: List[Path] = []
        for pattern in file_patterns:
            for candidate in self.project_path.rglob(pattern):
                if self._is_in_project_scope(candidate):
                    cpp_files.append(candidate)
        
        logger.info(f"Found {len(cpp_files)} C++ files to analyze")
        
        # Parse all files
        for cpp_file in cpp_files:
            try:
                self._parse_file(str(cpp_file))
            except Exception as e:
                logger.error(f"Failed to parse {cpp_file}: {e}")
        
        # Build CFGs for all functions
        for file_path, tu in self.translation_units.items():
            try:
                self._build_cfgs_from_tu(tu, file_path)
            except Exception as e:
                logger.error(f"Failed to build CFGs for {file_path}: {e}")
        
        # Extract call relationships
        self._extract_call_graph()
        
        # Identify leaf functions
        self._identify_leaf_functions()
        
        logger.info(
            f"Analysis complete: {len(self.translation_units)} files, "
            f"{len(self.function_cfgs)} functions, "
            f"{len(self.call_graph)} call relations"
        )
    
    def _parse_file(self, file_path: str) -> None:
        """Parse a single C++ file"""
        try:
            # Parse with common C++ flags
            args = [
                '-x', 'c++',
                '-std=c++17',
                '-I' + str(self.project_path),
            ]
            
            tu = self.index.parse(
                file_path,
                args=args,
                options=clang.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            if tu.diagnostics:
                errors = [d for d in tu.diagnostics if d.severity >= 3]
                if errors:
                    logger.warning(f"Errors parsing {file_path}: {len(errors)} errors")
            
            self.translation_units[file_path] = tu
            logger.debug(f"Parsed: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
    
    def _build_cfgs_from_tu(self, tu: clang.TranslationUnit, file_path: str) -> None:
        """Extract all function CFGs from a translation unit"""
        
        def visit_node(cursor: clang.Cursor, parent_qualified_name: str = ""):
            """Recursively visit AST nodes to find functions"""
            
            # HARD AST BOUNDARY: Only process nodes from project files
            if cursor.location.file:
                cursor_file_path = Path(cursor.location.file.name)
                # Use centralized exclusion check
                if not is_in_project_scope(cursor_file_path, self.project_path):
                    # Skip this node and all its children (external code)
                    return
            
            # Build qualified name for namespaces and classes
            if cursor.kind in [CursorKind.NAMESPACE, CursorKind.CLASS_DECL, 
                             CursorKind.STRUCT_DECL, CursorKind.CLASS_TEMPLATE]:
                if cursor.spelling:
                    qualified_prefix = f"{parent_qualified_name}::{cursor.spelling}" if parent_qualified_name else cursor.spelling
                else:
                    qualified_prefix = parent_qualified_name
            else:
                qualified_prefix = parent_qualified_name
            
            # Process function definitions
            if cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD]:
                if cursor.is_definition():
                    # Double-check file is in project scope
                    if cursor.location.file:
                        cursor_file_path = Path(cursor.location.file.name)
                        if is_in_project_scope(cursor_file_path, self.project_path):
                            try:
                                cfg = self._build_function_cfg(cursor, qualified_prefix, file_path)
                                if cfg:
                                    self.function_cfgs[cfg.qualified_name] = cfg
                                    logger.debug(f"Built CFG for: {cfg.qualified_name}")
                            except Exception as e:
                                logger.error(f"Failed to build CFG for {cursor.spelling}: {e}")
            
            # Recurse into children
            for child in cursor.get_children():
                visit_node(child, qualified_prefix)
        
        # Start traversal from root
        visit_node(tu.cursor)
    
    def _build_function_cfg(
        self, 
        cursor: clang.Cursor, 
        parent_qualified_name: str,
        file_path: str
    ) -> Optional[FunctionCFG]:
        """Build CFG for a single function"""
        
        if not cursor.is_definition():
            return None
        
        function_name = cursor.spelling
        qualified_name = f"{parent_qualified_name}::{function_name}" if parent_qualified_name else function_name
        
        # Extract parameters
        parameters = []
        for arg in cursor.get_arguments():
            param_name = arg.spelling
            param_type = arg.type.spelling
            parameters.append((param_name, param_type))
        
        # Extract return type
        return_type = cursor.result_type.spelling
        
        # Initialize CFG
        nodes: Dict[str, CFGNode] = {}
        node_counter = [0]  # Use list for mutability in nested functions
        
        def create_node(node_type: NodeType, cursor_ref: Optional[clang.Cursor] = None, text: str = "") -> str:
            """Helper to create a new CFG node"""
            node_id = f"{qualified_name}_node_{node_counter[0]}"
            node_counter[0] += 1
            
            source_range = None
            if cursor_ref and cursor_ref.extent:
                source_range = (
                    cursor_ref.extent.start.line,
                    cursor_ref.extent.end.line
                )
            
            nodes[node_id] = CFGNode(
                id=node_id,
                node_type=node_type,
                cursor=cursor_ref,
                source_range=source_range,
                text=text or (cursor_ref.spelling if cursor_ref else "")
            )
            return node_id
        
        # Create entry and exit nodes
        entry_node = create_node(NodeType.ENTRY, text=f"Entry: {function_name}")
        exit_nodes = []
        
        # Process function body
        body = None
        for child in cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body = child
                break
        
        if not body:
            # No body found
            exit_node = create_node(NodeType.EXIT, text=f"Exit: {function_name}")
            exit_nodes.append(exit_node)
            nodes[entry_node].successors.append(exit_node)
            nodes[exit_node].predecessors.append(entry_node)
        else:
            # Process statements in body
            current_nodes = [entry_node]
            exit_nodes = self._process_compound_stmt(
                body, 
                current_nodes, 
                nodes, 
                create_node, 
                qualified_name
            )
        
        # Create CFG object
        cfg = FunctionCFG(
            function_name=function_name,
            qualified_name=qualified_name,
            file_path=file_path,
            entry_node=entry_node,
            exit_nodes=exit_nodes,
            nodes=nodes,
            parameters=parameters,
            return_type=return_type
        )
        
        return cfg
    
    def _process_compound_stmt(
        self,
        compound: clang.Cursor,
        current_nodes: List[str],
        nodes: Dict[str, CFGNode],
        create_node,
        qualified_name: str
    ) -> List[str]:
        """Process a compound statement (block) and return exit nodes"""
        
        for stmt in compound.get_children():
            current_nodes = self._process_statement(
                stmt, current_nodes, nodes, create_node, qualified_name
            )
        
        return current_nodes
    
    def _process_statement(
        self,
        stmt: clang.Cursor,
        current_nodes: List[str],
        nodes: Dict[str, CFGNode],
        create_node,
        qualified_name: str
    ) -> List[str]:
        """Process a single statement and return next nodes"""
        
        kind = stmt.kind
        
        # If statement
        if kind == CursorKind.IF_STMT:
            return self._process_if_stmt(stmt, current_nodes, nodes, create_node, qualified_name)
        
        # While/For loops
        elif kind in [CursorKind.WHILE_STMT, CursorKind.FOR_STMT, CursorKind.DO_STMT]:
            return self._process_loop(stmt, current_nodes, nodes, create_node, qualified_name)
        
        # Return statement
        elif kind == CursorKind.RETURN_STMT:
            return_node = create_node(NodeType.RETURN, stmt, "return")
            for current in current_nodes:
                nodes[current].successors.append(return_node)
                nodes[return_node].predecessors.append(current)
            return []  # Return ends this path
        
        # Call expression
        elif kind == CursorKind.CALL_EXPR:
            call_node = create_node(NodeType.CALL, stmt)
            
            # Extract called function name
            callee = None
            for child in stmt.get_children():
                if child.kind in [CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR]:
                    callee = child.spelling
                    break
            
            if callee:
                nodes[call_node].called_functions.append(callee)
            
            for current in current_nodes:
                nodes[current].successors.append(call_node)
                nodes[call_node].predecessors.append(current)
            
            return [call_node]
        
        # Default: treat as simple statement
        else:
            stmt_node = create_node(NodeType.STATEMENT, stmt)
            
            # Check if this is a state mutation
            if self._is_state_mutation(stmt):
                nodes[stmt_node].mutates_state = True
            
            for current in current_nodes:
                nodes[current].successors.append(stmt_node)
                nodes[stmt_node].predecessors.append(current)
            
            return [stmt_node]
    
    def _process_if_stmt(
        self,
        if_stmt: clang.Cursor,
        current_nodes: List[str],
        nodes: Dict[str, CFGNode],
        create_node,
        qualified_name: str
    ) -> List[str]:
        """Process if statement"""
        
        children = list(if_stmt.get_children())
        
        # Create decision node
        condition = children[0] if children else None
        decision_node = create_node(
            NodeType.DECISION, 
            condition, 
            f"if ({condition.spelling if condition else '?'})"
        )
        nodes[decision_node].is_guard = True
        
        for current in current_nodes:
            nodes[current].successors.append(decision_node)
            nodes[decision_node].predecessors.append(current)
        
        # Process then branch
        then_branch = children[1] if len(children) > 1 else None
        then_exits = [decision_node]
        if then_branch:
            then_exits = self._process_statement(
                then_branch, [decision_node], nodes, create_node, qualified_name
            )
        
        # Process else branch
        else_branch = children[2] if len(children) > 2 else None
        else_exits = [decision_node]
        if else_branch:
            else_exits = self._process_statement(
                else_branch, [decision_node], nodes, create_node, qualified_name
            )
        
        # Merge branches
        return then_exits + else_exits
    
    def _process_loop(
        self,
        loop_stmt: clang.Cursor,
        current_nodes: List[str],
        nodes: Dict[str, CFGNode],
        create_node,
        qualified_name: str
    ) -> List[str]:
        """Process loop statement"""
        
        loop_node = create_node(NodeType.LOOP, loop_stmt, f"{loop_stmt.kind.name}")
        nodes[loop_node].is_guard = True
        
        for current in current_nodes:
            nodes[current].successors.append(loop_node)
            nodes[loop_node].predecessors.append(current)
        
        # Process loop body
        body = None
        for child in loop_stmt.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body = child
                break
        
        if body:
            body_exits = self._process_compound_stmt(
                body, [loop_node], nodes, create_node, qualified_name
            )
            
            # Loop back
            for exit_node in body_exits:
                nodes[exit_node].successors.append(loop_node)
                nodes[loop_node].predecessors.append(exit_node)
        
        # Loop can exit
        return [loop_node]
    
    def _is_state_mutation(self, cursor: clang.Cursor) -> bool:
        """Check if a statement mutates state"""
        
        # Assignment operators
        if cursor.kind in [
            CursorKind.BINARY_OPERATOR,
            CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            CursorKind.UNARY_OPERATOR
        ]:
            # Check for assignment-like operations
            tokens = list(cursor.get_tokens())
            for token in tokens:
                if token.spelling in ['=', '+=', '-=', '*=', '/=', '++', '--']:
                    return True
        
        # Member function calls (may mutate object state)
        if cursor.kind == CursorKind.CALL_EXPR:
            for child in cursor.get_children():
                if child.kind == CursorKind.MEMBER_REF_EXPR:
                    return True  # Conservative: assume member calls mutate
        
        return False
    
    def _extract_call_graph(self) -> None:
        """Extract call relationships from all functions"""
        
        for qualified_name, cfg in self.function_cfgs.items():
            for node_id, node in cfg.nodes.items():
                if node.node_type == NodeType.CALL and node.called_functions:
                    for callee in node.called_functions:
                        # Determine if call is conditional
                        is_conditional = any(
                            pred_node.node_type in [NodeType.DECISION, NodeType.LOOP]
                            for pred_id in node.predecessors
                            for pred_node in [cfg.nodes.get(pred_id)]
                            if pred_node
                        )
                        
                        relation = CallRelation(
                            caller=qualified_name,
                            callee=callee,
                            call_site=f"{cfg.file_path}:{node.source_range[0] if node.source_range else '?'}",
                            is_conditional=is_conditional
                        )
                        self.call_graph.append(relation)
    
    def _identify_leaf_functions(self) -> None:
        """Identify functions that don't call other functions"""
        # Only consider calls where both caller and callee are project functions.
        project_functions = set(self.function_cfgs.keys())
        project_callers: Set[str] = set()

        for rel in self.call_graph:
            if rel.caller in project_functions and rel.callee in project_functions:
                project_callers.add(rel.caller)

        for qualified_name, cfg in self.function_cfgs.items():
            # Leaf = does not call any other project-defined function
            cfg.is_leaf = qualified_name not in project_callers
            if cfg.is_leaf:
                logger.debug(f"Identified leaf function: {qualified_name}")
    
    def get_function_cfg(self, qualified_name: str) -> Optional[FunctionCFG]:
        """Retrieve CFG for a specific function"""
        return self.function_cfgs.get(qualified_name)
    
    def get_leaf_functions(self) -> List[FunctionCFG]:
        """Get all leaf functions (functions that don't call others)"""
        return [cfg for cfg in self.function_cfgs.values() if cfg.is_leaf]
    
    def get_callers(self, function_name: str) -> List[str]:
        """Get all functions that call the specified function"""
        return [
            rel.caller 
            for rel in self.call_graph 
            if rel.callee == function_name
        ]
    
    def get_callees(self, function_name: str) -> List[str]:
        """Get all functions called by the specified function"""
        return [
            rel.callee 
            for rel in self.call_graph 
            if rel.caller == function_name
        ]




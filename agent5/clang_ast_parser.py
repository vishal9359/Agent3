"""
Clang AST Parser Module - Stage 1

This module provides full AST construction using libclang bindings for Python.
It extracts:
- Complete AST for all translation units
- Control-flow graphs (CFG) per function
- Call relationships (for understanding, not visualization)
- Leaf-level execution units (basic blocks)
- Guard conditions, state mutations, error exits

NO LLM is used in this stage. Pure deterministic analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

try:
    import clang.cindex as clang
    from clang.cindex import CursorKind, TypeKind
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    logging.warning("libclang not available - Clang AST features disabled")

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of AST/CFG nodes"""
    ENTRY = "entry"
    EXIT = "exit"
    STATEMENT = "statement"
    DECISION = "decision"
    LOOP = "loop"
    CALL = "call"
    RETURN = "return"
    ERROR_EXIT = "error_exit"
    EARLY_EXIT = "early_exit"


@dataclass
class ASTNode:
    """
    Wrapper around Clang's Cursor with additional metadata.
    
    Represents a node in the Abstract Syntax Tree with semantic annotations.
    """
    cursor: Optional[Any] = None  # clang.Cursor when available
    node_type: Optional[NodeType] = None
    kind: Optional[Any] = None  # CursorKind when available
    
    @property
    def children(self) -> List['ASTNode']:
        """Get children of this AST node"""
        if not self.cursor:
            return []
        return [ASTNode(cursor=child, kind=child.kind) for child in self.cursor.get_children()]


@dataclass
class BasicBlock:
    """Represents a basic block in the CFG."""
    id: str
    statements: List[str] = field(default_factory=list)
    has_condition: bool = False
    condition_expr: Optional[str] = None
    successors: List[str] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    is_leaf: bool = False  # True if no further function calls


@dataclass
class FunctionCFG:
    """Control Flow Graph for a single function."""
    function_name: str
    file_path: str
    entry_block: str
    exit_blocks: List[str] = field(default_factory=list)
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    local_vars: Set[str] = field(default_factory=set)
    params: List[str] = field(default_factory=list)
    return_type: str = ""
    is_leaf_function: bool = False  # True if makes no calls


@dataclass
class CallRelationship:
    """Represents a call from one function to another."""
    caller: str
    callee: str
    call_site_file: str
    call_site_line: int
    is_conditional: bool = False
    context: str = ""  # Surrounding code context


@dataclass
class ProjectAST:
    """Complete AST representation of a C++ project."""
    project_path: Path
    translation_units: List[str] = field(default_factory=list)
    function_cfgs: Dict[str, FunctionCFG] = field(default_factory=dict)
    call_graph: List[CallRelationship] = field(default_factory=list)
    entry_points: Set[str] = field(default_factory=set)
    leaf_functions: Set[str] = field(default_factory=set)


class ClangASTParser:
    """
    Clang-based AST parser for C++ projects.
    
    This parser provides deterministic, LLM-free analysis of C++ codebases,
    extracting semantic information at the AST and CFG level.
    """

    def __init__(self, project_path: Path, compile_commands: Optional[Path] = None):
        """
        Initialize the Clang AST parser.
        
        Args:
            project_path: Root path of the C++ project
            compile_commands: Optional path to compile_commands.json
        """
        if not CLANG_AVAILABLE:
            raise RuntimeError(
                "libclang is not available. Install it via:\n"
                "  pip install libclang\n"
                "or ensure clang is installed on your system."
            )
        
        self.project_path = project_path
        self.compile_commands = compile_commands
        self.index = clang.Index.create()
        self.ast = ProjectAST(project_path=project_path)
        
        # Configure clang
        self._init_clang()
    
    def _init_clang(self):
        """Initialize clang configuration."""
        try:
            clang.Config.set_library_file(self._find_libclang())
        except Exception as e:
            logger.warning(f"Could not configure libclang: {e}")
    
    def _find_libclang(self) -> Optional[str]:
        """Attempt to locate libclang library."""
        # Common locations
        import platform
        system = platform.system()
        
        if system == "Windows":
            candidates = [
                "C:\\Program Files\\LLVM\\bin\\libclang.dll",
                "C:\\Program Files (x86)\\LLVM\\bin\\libclang.dll",
            ]
        elif system == "Linux":
            candidates = [
                "/usr/lib/llvm-14/lib/libclang.so",
                "/usr/lib/llvm-13/lib/libclang.so",
                "/usr/lib/x86_64-linux-gnu/libclang.so",
            ]
        else:  # macOS
            candidates = [
                "/usr/local/opt/llvm/lib/libclang.dylib",
                "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
            ]
        
        for path in candidates:
            if Path(path).exists():
                return path
        
        return None
    
    def parse_project(self) -> ProjectAST:
        """
        Parse the entire C++ project.
        
        Returns:
            ProjectAST containing all extracted information
        """
        logger.info(f"Parsing C++ project at {self.project_path}")
        
        # Find all C++ source files
        cpp_files = self._find_cpp_files()
        
        if not cpp_files:
            logger.warning(f"No C++ files found in {self.project_path}")
            return self.ast
        
        logger.info(f"Found {len(cpp_files)} C++ files")
        
        # Parse each translation unit
        for cpp_file in cpp_files:
            try:
                self._parse_translation_unit(cpp_file)
            except Exception as e:
                logger.error(f"Failed to parse {cpp_file}: {e}")
        
        # Build call graph
        self._build_call_graph()
        
        # Identify leaf functions
        self._identify_leaf_functions()
        
        # Identify entry points
        self._identify_entry_points()
        
        logger.info(f"Parsed {len(self.ast.function_cfgs)} functions")
        logger.info(f"Found {len(self.ast.leaf_functions)} leaf functions")
        logger.info(f"Found {len(self.ast.entry_points)} potential entry points")
        
        return self.ast
    
    def _find_cpp_files(self) -> List[Path]:
        """Find all C++ source files in the project."""
        extensions = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h", ".hh"}
        cpp_files = []
        
        for ext in extensions:
            cpp_files.extend(self.project_path.rglob(f"*{ext}"))
        
        return cpp_files
    
    def _parse_translation_unit(self, file_path: Path):
        """Parse a single translation unit (source file)."""
        logger.debug(f"Parsing translation unit: {file_path}")
        
        # Parse with clang
        tu = self.index.parse(
            str(file_path),
            args=["-std=c++17", "-I" + str(self.project_path)],
            options=clang.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        
        if not tu:
            logger.error(f"Failed to parse {file_path}")
            return
        
        self.ast.translation_units.append(str(file_path))
        
        # Extract functions and build CFGs
        self._extract_functions(tu.cursor, str(file_path))
    
    def _extract_functions(self, cursor: "clang.Cursor", file_path: str):
        """Recursively extract all function definitions."""
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            self._build_function_cfg(cursor, file_path)
        
        # Recurse into children
        for child in cursor.get_children():
            self._extract_functions(child, file_path)
    
    def _build_function_cfg(self, func_cursor: "clang.Cursor", file_path: str):
        """Build a CFG for a single function."""
        func_name = func_cursor.spelling
        qualified_name = self._get_qualified_name(func_cursor)
        
        logger.debug(f"Building CFG for function: {qualified_name}")
        
        # Create CFG
        cfg = FunctionCFG(
            function_name=qualified_name,
            file_path=file_path,
            entry_block="entry",
            return_type=func_cursor.result_type.spelling,
        )
        
        # Extract parameters
        for param in func_cursor.get_arguments():
            cfg.params.append(param.spelling)
        
        # Build basic blocks from function body
        block_id_counter = [0]  # Mutable counter
        
        def next_block_id():
            block_id_counter[0] += 1
            return f"block_{block_id_counter[0]}"
        
        # Create entry block
        entry_block = BasicBlock(id="entry")
        cfg.blocks["entry"] = entry_block
        
        # Process function body
        body = self._get_function_body(func_cursor)
        if body:
            self._process_statement_block(
                body, cfg, "entry", next_block_id, is_root=True
            )
        
        # Identify exit blocks
        for block_id, block in cfg.blocks.items():
            if not block.successors:
                cfg.exit_blocks.append(block_id)
        
        # Store CFG
        self.ast.function_cfgs[qualified_name] = cfg
    
    def _get_qualified_name(self, cursor: "clang.Cursor") -> str:
        """Get fully qualified name of a function."""
        parts = []
        current = cursor
        
        while current and current.kind != CursorKind.TRANSLATION_UNIT:
            if current.kind in {
                CursorKind.NAMESPACE,
                CursorKind.CLASS_DECL,
                CursorKind.STRUCT_DECL,
                CursorKind.FUNCTION_DECL,
            }:
                parts.append(current.spelling)
            current = current.semantic_parent
        
        parts.reverse()
        return "::".join(parts) if parts else cursor.spelling
    
    def _get_function_body(self, func_cursor: "clang.Cursor") -> Optional["clang.Cursor"]:
        """Get the compound statement representing the function body."""
        for child in func_cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                return child
        return None
    
    def _process_statement_block(
        self,
        stmt_cursor: "clang.Cursor",
        cfg: FunctionCFG,
        current_block_id: str,
        next_block_id,
        is_root: bool = False,
    ) -> str:
        """
        Process a statement block and build CFG nodes.
        
        Returns the ID of the last block created.
        """
        current_block = cfg.blocks[current_block_id]
        
        for child in stmt_cursor.get_children():
            if child.kind == CursorKind.IF_STMT:
                # Create decision node
                cond_block_id = next_block_id()
                cond_block = BasicBlock(
                    id=cond_block_id,
                    has_condition=True,
                    condition_expr=self._get_source_text(child),
                )
                cfg.blocks[cond_block_id] = cond_block
                current_block.successors.append(cond_block_id)
                cond_block.predecessors.append(current_block_id)
                
                # Process branches (simplified)
                current_block_id = cond_block_id
                
            elif child.kind == CursorKind.RETURN_STMT:
                # Return statement
                return_text = self._get_source_text(child)
                current_block.statements.append(return_text)
                
            elif child.kind == CursorKind.CALL_EXPR:
                # Function call
                call_text = self._get_source_text(child)
                current_block.statements.append(call_text)
                
            elif child.kind in {
                CursorKind.VAR_DECL,
                CursorKind.DECL_STMT,
            }:
                # Variable declaration
                decl_text = self._get_source_text(child)
                current_block.statements.append(decl_text)
                
                if child.kind == CursorKind.VAR_DECL:
                    cfg.local_vars.add(child.spelling)
            
            else:
                # Other statements
                stmt_text = self._get_source_text(child)
                if stmt_text:
                    current_block.statements.append(stmt_text)
        
        return current_block_id
    
    def _get_source_text(self, cursor: "clang.Cursor") -> str:
        """Get source code text for a cursor."""
        try:
            extent = cursor.extent
            with open(extent.start.file.name, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                
            start_line = extent.start.line - 1
            end_line = extent.end.line - 1
            
            if start_line == end_line:
                line = lines[start_line]
                return line[extent.start.column - 1 : extent.end.column - 1].strip()
            else:
                # Multi-line (simplified)
                return " ".join(
                    lines[start_line:end_line + 1]
                ).strip()
        except Exception:
            return cursor.spelling or ""
    
    def _build_call_graph(self):
        """Build the call graph from all functions."""
        logger.debug("Building call graph")
        
        for func_name, cfg in self.ast.function_cfgs.items():
            for block in cfg.blocks.values():
                for stmt in block.statements:
                    # Simple pattern matching for function calls
                    # (In production, use AST traversal)
                    if "(" in stmt and ")" in stmt:
                        callee = self._extract_callee_name(stmt)
                        if callee and callee in self.ast.function_cfgs:
                            rel = CallRelationship(
                                caller=func_name,
                                callee=callee,
                                call_site_file=cfg.file_path,
                                call_site_line=0,  # TODO: extract line
                                is_conditional=block.has_condition,
                                context=stmt,
                            )
                            self.ast.call_graph.append(rel)
    
    def _extract_callee_name(self, stmt: str) -> Optional[str]:
        """Extract function name from a call statement (simplified)."""
        # This is a simplified heuristic
        # In production, use AST traversal
        import re
        match = re.search(r"(\w+)\s*\(", stmt)
        if match:
            return match.group(1)
        return None
    
    def _identify_leaf_functions(self):
        """Identify functions that make no calls (leaf functions)."""
        callers = {rel.caller for rel in self.ast.call_graph}
        all_functions = set(self.ast.function_cfgs.keys())
        
        self.ast.leaf_functions = all_functions - callers
        
        # Mark in CFGs
        for func_name in self.ast.leaf_functions:
            if func_name in self.ast.function_cfgs:
                self.ast.function_cfgs[func_name].is_leaf_function = True
                
                # Mark all blocks as leaf blocks
                for block in self.ast.function_cfgs[func_name].blocks.values():
                    block.is_leaf = True
    
    def _identify_entry_points(self):
        """Identify potential entry point functions."""
        # Entry points are functions that:
        # 1. Are named 'main'
        # 2. Have no callers within the project
        # 3. Match common entry point patterns
        
        callees = {rel.callee for rel in self.ast.call_graph}
        all_functions = set(self.ast.function_cfgs.keys())
        
        # Functions with no callers
        no_callers = all_functions - callees
        
        for func_name in no_callers:
            # Add common entry point patterns
            lower_name = func_name.lower()
            if any(
                pattern in lower_name
                for pattern in ["main", "handle", "process", "execute", "run", "start"]
            ):
                self.ast.entry_points.add(func_name)
    
    def get_function_cfg(self, function_name: str) -> Optional[FunctionCFG]:
        """Get the CFG for a specific function."""
        return self.ast.function_cfgs.get(function_name)
    
    def get_call_chain(self, entry_function: str) -> List[str]:
        """Get all functions reachable from an entry function."""
        visited = set()
        stack = [entry_function]
        chain = []
        
        while stack:
            func = stack.pop()
            if func in visited:
                continue
            
            visited.add(func)
            chain.append(func)
            
            # Add callees
            for rel in self.ast.call_graph:
                if rel.caller == func and rel.callee not in visited:
                    stack.append(rel.callee)
        
        return chain


def parse_cpp_project(project_path: Path) -> ProjectAST:
    """
    Convenience function to parse a C++ project.
    
    Args:
        project_path: Root path of the C++ project
    
    Returns:
        ProjectAST containing full AST and CFG information
    """
    parser = ClangASTParser(project_path)
    return parser.parse_project()


# Re-export FunctionInfo and CallGraph from call_graph_builder for backward compatibility
try:
    from agent5.call_graph_builder import FunctionInfo, CallGraph
except (ImportError, ModuleNotFoundError) as e:
    # If call_graph_builder is not available, define minimal stubs
    logger.warning(f"call_graph_builder not available ({e}), defining minimal stubs for FunctionInfo and CallGraph")
    
    @dataclass
    class FunctionInfo:  # type: ignore
        """Minimal stub for FunctionInfo when call_graph_builder is not available."""
        name: str = ""
        file_path: Any = None
        namespace: Optional[str] = None
        class_name: Optional[str] = None
        qualified_name: Optional[str] = None
        start_line: int = 0
        end_line: int = 0
        is_leaf: bool = False
        calls: List[str] = field(default_factory=list)
        called_by: List[str] = field(default_factory=list)
        body_statements: Optional[List[Any]] = None
        ast_root: Optional[Any] = None
        metadata: Dict[str, Any] = field(default_factory=dict)
    
    @dataclass
    class CallGraph:  # type: ignore
        """Minimal stub for CallGraph when call_graph_builder is not available."""
        nodes: Dict[str, Any] = field(default_factory=dict)
        edges: List[Tuple[str, str]] = field(default_factory=list)
        functions: Dict[str, FunctionInfo] = field(default_factory=dict)
        
        def get_leaf_nodes(self) -> List[Any]:
            return []
        
        def get_nodes_at_level(self, level: int) -> List[Any]:
            return []
        
        def get_callees(self, func_name: str) -> List[Any]:
            return []
        
        def compute_levels(self) -> None:
            pass

"""
Stage 1: Full AST Construction using Clang.

This module parses the entire C++ project using libclang to extract:
- Abstract Syntax Tree (AST) for all translation units
- Control Flow Graphs (CFG) per function
- Call relationships (for understanding, not visualization)
- Leaf-level execution units (basic blocks)
- Guard conditions, state mutations, error exits

NO LLM involvement at this stage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import clang.cindex
from clang.cindex import (
    Config,
    Cursor,
    CursorKind,
    Index,
    TranslationUnit,
    TypeKind,
)

logger = logging.getLogger(__name__)


@dataclass
class BasicBlock:
    """
    A basic block is a sequence of statements with:
    - Single entry point
    - Single exit point
    - No branches except at the end
    """
    
    id: str
    statements: list[Cursor] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    is_entry: bool = False
    is_exit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlFlowGraph:
    """Control Flow Graph for a single function."""
    
    function_name: str
    function_cursor: Cursor
    entry_block: str
    exit_blocks: list[str] = field(default_factory=list)
    basic_blocks: dict[str, BasicBlock] = field(default_factory=dict)
    
    def get_leaf_blocks(self) -> list[BasicBlock]:
        """Return basic blocks that don't call other functions."""
        leaf_blocks = []
        for block in self.basic_blocks.values():
            has_call = False
            for stmt in block.statements:
                if self._contains_call(stmt):
                    has_call = True
                    break
            if not has_call:
                leaf_blocks.append(block)
        return leaf_blocks
    
    def _contains_call(self, cursor: Cursor) -> bool:
        """Check if cursor contains a function call."""
        if cursor.kind == CursorKind.CALL_EXPR:
            return True
        for child in cursor.get_children():
            if self._contains_call(child):
                return True
        return False


@dataclass
class CallRelationship:
    """Represents a call relationship between functions."""
    
    caller: str  # Function name
    callee: str  # Function name
    call_site: Cursor  # Location of the call
    caller_cursor: Cursor  # Cursor to caller function
    callee_cursor: Cursor | None = None  # Cursor to callee function (if available)


@dataclass
class ProjectAST:
    """Complete AST representation of the C++ project."""
    
    project_path: Path
    translation_units: dict[str, TranslationUnit] = field(default_factory=dict)
    functions: dict[str, Cursor] = field(default_factory=dict)  # function_name -> cursor
    cfgs: dict[str, ControlFlowGraph] = field(default_factory=dict)  # function_name -> CFG
    call_graph: list[CallRelationship] = field(default_factory=list)
    
    def get_leaf_functions(self) -> list[str]:
        """Return functions that don't call other project functions."""
        callers = {rel.caller for rel in self.call_graph}
        return [name for name in self.functions.keys() if name not in callers]
    
    def get_callers(self, function_name: str) -> list[str]:
        """Get all functions that call the given function."""
        return [rel.caller for rel in self.call_graph if rel.callee == function_name]
    
    def get_callees(self, function_name: str) -> list[CallRelationship]:
        """Get all functions called by the given function."""
        return [rel for rel in self.call_graph if rel.caller == function_name]


class ClangASTExtractor:
    """
    Stage 1: Full AST Construction.
    
    Parses C++ project using Clang and builds:
    - AST for all translation units
    - CFG per function
    - Call relationships
    """
    
    def __init__(self, project_path: Path, include_paths: list[Path] | None = None):
        """
        Initialize Clang AST extractor.
        
        Args:
            project_path: Root path of the C++ project
            include_paths: Additional include directories
        """
        self.project_path = Path(project_path).resolve()
        if not self.project_path.is_dir():
            raise ValueError(f"Invalid project root (not a directory): {self.project_path}")
        self.include_paths = include_paths or []
        
        # Initialize libclang
        try:
            self.index = Index.create()
        except Exception as e:
            logger.error(f"Failed to initialize libclang: {e}")
            raise
        
        logger.info(f"Initialized Clang AST extractor for project: {project_path}")
    
    def extract_project_ast(self, cpp_files: list[Path] | None = None) -> ProjectAST:
        """
        Extract complete AST for the project.
        
        Args:
            cpp_files: List of C++ files to parse. If None, discover all .cpp/.cc files
        
        Returns:
            ProjectAST containing all extracted information
        """
        if cpp_files is None:
            cpp_files = self._discover_cpp_files()
        
        logger.info(f"Parsing {len(cpp_files)} C++ files...")
        
        project_ast = ProjectAST(project_path=self.project_path)
        
        # Parse all translation units
        for cpp_file in cpp_files:
            try:
                tu = self._parse_file(cpp_file)
                if tu:
                    project_ast.translation_units[str(cpp_file)] = tu
                    self._extract_functions(tu, project_ast)
            except Exception as e:
                logger.warning(f"Failed to parse {cpp_file}: {e}")
        
        logger.info(f"Extracted {len(project_ast.functions)} functions")
        
        # Build CFGs for all functions
        logger.info("Building Control Flow Graphs...")
        for func_name, cursor in project_ast.functions.items():
            try:
                cfg = self._build_cfg(cursor)
                project_ast.cfgs[func_name] = cfg
            except Exception as e:
                logger.warning(f"Failed to build CFG for {func_name}: {e}")
        
        # Extract call relationships
        logger.info("Extracting call relationships...")
        self._extract_call_relationships(project_ast)
        
        logger.info(f"Extracted {len(project_ast.call_graph)} call relationships")
        
        return project_ast
    
    def _discover_cpp_files(self) -> list[Path]:
        """Discover all C++ source files in the project, enforcing project boundary."""
        extensions = {".cpp", ".cc", ".cxx", ".c++"}
        cpp_files: list[Path] = []

        excluded = {"build", "out", ".cache", "external", "third_party"}

        def _in_scope(path: Path) -> bool:
            try:
                path = path.resolve()
            except Exception:
                return False
            if self.project_path not in path.parents and path != self.project_path:
                return False
            for part in path.parts:
                if part in excluded or part.startswith("bazel-"):
                    return False
            return True
        
        for ext in extensions:
            for candidate in self.project_path.rglob(f"*{ext}"):
                if _in_scope(candidate):
                    cpp_files.append(candidate)
        
        return sorted(cpp_files)
    
    def _parse_file(self, file_path: Path) -> TranslationUnit | None:
        """Parse a single C++ file into a translation unit."""
        args = ["-std=c++17"]
        
        # Add include paths
        for include_path in self.include_paths:
            args.append(f"-I{include_path}")
        
        try:
            tu = self.index.parse(
                str(file_path),
                args=args,
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            
            # Check for errors
            if tu.diagnostics:
                errors = [d for d in tu.diagnostics if d.severity >= 3]
                if errors:
                    logger.warning(f"Parse errors in {file_path}: {len(errors)} errors")
            
            return tu
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            return None
    
    def _extract_functions(self, tu: TranslationUnit, project_ast: ProjectAST) -> None:
        """Extract all functions from a translation unit."""
        
        def visit(cursor: Cursor) -> None:
            # Only consider cursors from the main file (not includes)
            if cursor.location.file and str(cursor.location.file.name).startswith(str(self.project_path)):
                if cursor.kind == CursorKind.FUNCTION_DECL:
                    # Only include definitions, not declarations
                    if cursor.is_definition():
                        func_name = self._get_qualified_name(cursor)
                        project_ast.functions[func_name] = cursor
                        logger.debug(f"Found function: {func_name}")
                
                elif cursor.kind == CursorKind.CXX_METHOD:
                    if cursor.is_definition():
                        func_name = self._get_qualified_name(cursor)
                        project_ast.functions[func_name] = cursor
                        logger.debug(f"Found method: {func_name}")
            
            # Recurse into children
            for child in cursor.get_children():
                visit(child)
        
        visit(tu.cursor)
    
    def _get_qualified_name(self, cursor: Cursor) -> str:
        """Get the fully qualified name of a function/method."""
        parts = []
        
        current = cursor
        while current:
            if current.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD):
                parts.append(current.spelling)
            elif current.kind in (CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, 
                                 CursorKind.NAMESPACE):
                parts.append(current.spelling)
            current = current.semantic_parent
        
        return "::".join(reversed(parts))
    
    def _build_cfg(self, function_cursor: Cursor) -> ControlFlowGraph:
        """
        Build a Control Flow Graph for a function.
        
        This is a simplified CFG that identifies basic blocks and their relationships.
        """
        func_name = self._get_qualified_name(function_cursor)
        cfg = ControlFlowGraph(
            function_name=func_name,
            function_cursor=function_cursor
        )
        
        # Find function body
        body = None
        for child in function_cursor.get_children():
            if child.kind == CursorKind.COMPOUND_STMT:
                body = child
                break
        
        if not body:
            # No body (declaration only)
            entry = BasicBlock(id="entry", is_entry=True, is_exit=True)
            cfg.basic_blocks["entry"] = entry
            cfg.entry_block = "entry"
            cfg.exit_blocks = ["entry"]
            return cfg
        
        # Build basic blocks from statements
        block_id = 0
        current_block = BasicBlock(id=f"bb{block_id}", is_entry=True)
        cfg.entry_block = current_block.id
        
        for stmt in body.get_children():
            if self._is_control_flow_stmt(stmt):
                # Save current block
                if current_block.statements or current_block.is_entry:
                    cfg.basic_blocks[current_block.id] = current_block
                
                # Create new block for control flow
                block_id += 1
                new_block = BasicBlock(id=f"bb{block_id}")
                new_block.statements.append(stmt)
                cfg.basic_blocks[new_block.id] = new_block
                
                # Link blocks
                current_block.successors.append(new_block.id)
                new_block.predecessors.append(current_block.id)
                
                # Start new block after control flow
                block_id += 1
                current_block = BasicBlock(id=f"bb{block_id}")
                new_block.successors.append(current_block.id)
                current_block.predecessors.append(new_block.id)
            else:
                # Regular statement
                current_block.statements.append(stmt)
        
        # Save final block
        if current_block.statements or current_block.is_entry:
            current_block.is_exit = True
            cfg.basic_blocks[current_block.id] = current_block
            cfg.exit_blocks.append(current_block.id)
        
        return cfg
    
    def _is_control_flow_stmt(self, cursor: Cursor) -> bool:
        """Check if a statement affects control flow."""
        return cursor.kind in {
            CursorKind.IF_STMT,
            CursorKind.WHILE_STMT,
            CursorKind.FOR_STMT,
            CursorKind.DO_STMT,
            CursorKind.SWITCH_STMT,
            CursorKind.RETURN_STMT,
            CursorKind.BREAK_STMT,
            CursorKind.CONTINUE_STMT,
            CursorKind.CXX_THROW_EXPR,
        }
    
    def _extract_call_relationships(self, project_ast: ProjectAST) -> None:
        """Extract call relationships between functions."""
        
        for func_name, func_cursor in project_ast.functions.items():
            self._find_calls_in_function(func_cursor, func_name, project_ast)
    
    def _find_calls_in_function(
        self,
        cursor: Cursor,
        caller_name: str,
        project_ast: ProjectAST
    ) -> None:
        """Find all function calls in a function."""
        
        if cursor.kind == CursorKind.CALL_EXPR:
            # Get the called function
            callee = None
            for child in cursor.get_children():
                if child.kind in (CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR):
                    callee = child.referenced
                    break
            
            if callee and callee.kind in (CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD):
                callee_name = self._get_qualified_name(callee)
                
                # Only track calls to project functions
                if callee_name in project_ast.functions:
                    rel = CallRelationship(
                        caller=caller_name,
                        callee=callee_name,
                        call_site=cursor,
                        caller_cursor=project_ast.functions[caller_name],
                        callee_cursor=project_ast.functions.get(callee_name)
                    )
                    project_ast.call_graph.append(rel)
        
        # Recurse into children
        for child in cursor.get_children():
            self._find_calls_in_function(child, caller_name, project_ast)


def extract_ast_for_project(
    project_path: Path,
    include_paths: list[Path] | None = None,
    cpp_files: list[Path] | None = None
) -> ProjectAST:
    """
    Convenience function to extract AST for a C++ project.
    
    Args:
        project_path: Root path of the project
        include_paths: Additional include directories
        cpp_files: Specific files to parse (or None for all)
    
    Returns:
        ProjectAST containing complete AST information
    """
    extractor = ClangASTExtractor(project_path, include_paths)
    return extractor.extract_project_ast(cpp_files)

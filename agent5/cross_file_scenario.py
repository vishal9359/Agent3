"""
Cross-file scenario extraction for C++ projects.

This module implements scenario flow extraction that follows function calls across
multiple files in a project, providing a complete view of the execution flow.

Key Principles:
- Start from entry point in entry file
- Follow function calls based on detail level
- Search for function definitions across the project
- Integrate SFMs from multiple files
- Limit depth to avoid infinite recursion
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language

from agent5.logging_utils import log_info, log_warning
from agent5.scenario_extractor import (
    DetailLevel,
    ScenarioFlowModel,
    SFMBuilder,
    SFMNode,
    extract_scenario_from_function,
)


@dataclass
class FunctionCallSite:
    """Represents a function call encountered during scenario extraction."""
    
    callee_name: str
    caller_context: str  # Context where the call happens
    call_site_node_id: str | None = None  # SFM node ID where this call occurs
    classification: str = "unknown"  # business, validation, state, critical, utility


@dataclass
class CrossFileScenarioExtractor:
    """
    Extracts scenarios across multiple files in a C++ project.
    
    Strategy:
    1. Start with entry function in entry file
    2. Build SFM for that function
    3. Identify function calls that should be expanded (based on detail level)
    4. Search for function definitions across project
    5. Recursively integrate their SFMs (with depth limit)
    """
    
    project_path: Path
    detail_level: DetailLevel = DetailLevel.MEDIUM
    max_steps: int = 50
    max_depth: int = 3  # Maximum recursion depth for following calls
    _visited_functions: set[str] = field(default_factory=set)
    _function_cache: dict[str, tuple[Path, str]] = field(default_factory=dict)  # name -> (file, code)
    
    def extract_cross_file_scenario(
        self,
        entry_file: Path,
        entry_function: str,
    ) -> ScenarioFlowModel:
        """
        Extract a scenario flow that spans multiple files.
        
        Args:
            entry_file: File containing the entry function
            entry_function: Name of the entry function
            
        Returns:
            ScenarioFlowModel spanning multiple files
            
        Raises:
            RuntimeError: If entry function cannot be found or SFM cannot be built
        """
        log_info(f"Starting cross-file scenario extraction from {entry_function} in {entry_file.name}")
        log_info(f"Project path: {self.project_path}")
        log_info(f"Detail level: {self.detail_level.value}, Max depth: {self.max_depth}")
        
        # Build index of functions in the project
        self._build_function_index()
        
        # Extract the main scenario from the entry function
        entry_code = entry_file.read_text(encoding="utf-8", errors="ignore")
        sfm = extract_scenario_from_function(
            entry_code,
            function_name=entry_function,
            max_steps=self.max_steps,
            detail_level=self.detail_level,
        )
        
        # Mark entry function as visited
        self._visited_functions.add(entry_function)
        
        # Expand function calls if depth > 0
        if self.max_depth > 0:
            sfm = self._expand_function_calls(sfm, depth=0)
        
        log_info(f"Cross-file scenario extraction complete: {len(sfm.nodes)} nodes, {len(sfm.edges)} edges")
        return sfm
    
    def _build_function_index(self) -> None:
        """
        Build an index of all functions in the project.
        
        Scans all C++ files in the project and creates a mapping of
        function names to their file paths and source code.
        """
        log_info("Building function index...")
        
        # Find all C++ files in project
        cpp_extensions = {".cpp", ".cc", ".cxx", ".c", ".hpp", ".h", ".hxx"}
        cpp_files = []
        
        for ext in cpp_extensions:
            cpp_files.extend(self.project_path.rglob(f"*{ext}"))
        
        log_info(f"Found {len(cpp_files)} C++ files")
        
        # Parse each file and extract function names
        parser = Parser()
        _set_parser_language(parser)
        
        for file_path in cpp_files:
            try:
                source_code = file_path.read_text(encoding="utf-8", errors="ignore")
                source_bytes = source_code.encode("utf-8", errors="ignore")
                
                tree = parser.parse(source_bytes)
                root = tree.root_node
                
                # Find all function definitions
                self._index_functions_in_tree(source_bytes, root, file_path, source_code)
            except Exception as e:
                log_warning(f"Failed to index {file_path.name}: {e}")
        
        log_info(f"Indexed {len(self._function_cache)} functions")
    
    def _index_functions_in_tree(
        self,
        source_bytes: bytes,
        node: Node,
        file_path: Path,
        source_code: str,
    ) -> None:
        """Recursively index all function definitions in the AST."""
        if node.type in {"function_definition", "constructor_or_destructor_definition"}:
            # Extract function name
            func_name = self._get_function_name(source_bytes, node)
            if func_name:
                # Store the function
                if func_name not in self._function_cache:
                    self._function_cache[func_name] = (file_path, source_code)
        
        # Recurse into children
        for child in node.children:
            self._index_functions_in_tree(source_bytes, child, file_path, source_code)
    
    def _get_function_name(self, source_bytes: bytes, node: Node) -> str | None:
        """Extract function name from a function definition node."""
        # Try the declarator field
        decl = node.child_by_field_name("declarator")
        if decl:
            return self._extract_identifier(source_bytes, decl)
        
        # Try the name field
        name_node = node.child_by_field_name("name")
        if name_node:
            return _node_text(source_bytes, name_node).strip()
        
        return None
    
    def _extract_identifier(self, source_bytes: bytes, node: Node) -> str | None:
        """Extract identifier from a declarator node."""
        if node.type in {"identifier", "field_identifier", "type_identifier"}:
            return _node_text(source_bytes, node).strip()
        
        # Traverse to find identifier
        for child in node.children:
            result = self._extract_identifier(source_bytes, child)
            if result:
                return result
        
        return None
    
    def _expand_function_calls(
        self,
        sfm: ScenarioFlowModel,
        depth: int,
    ) -> ScenarioFlowModel:
        """
        Expand function calls in the SFM by following them to their definitions.
        
        Args:
            sfm: The scenario flow model to expand
            depth: Current recursion depth
            
        Returns:
            Expanded ScenarioFlowModel
        """
        if depth >= self.max_depth:
            return sfm
        
        # Identify function calls in the SFM
        function_calls = self._identify_function_calls(sfm)
        
        if not function_calls:
            return sfm
        
        log_info(f"Found {len(function_calls)} function calls at depth {depth}")
        
        # Expand each function call
        for call in function_calls:
            if call.callee_name in self._visited_functions:
                continue
            
            # Find the function definition
            if call.callee_name not in self._function_cache:
                log_warning(f"Cannot find definition for {call.callee_name}")
                continue
            
            file_path, source_code = self._function_cache[call.callee_name]
            log_info(f"Expanding {call.callee_name} from {file_path.name}")
            
            # Extract SFM for the called function
            try:
                called_sfm = extract_scenario_from_function(
                    source_code,
                    function_name=call.callee_name,
                    max_steps=self.max_steps,
                    detail_level=self.detail_level,
                )
                
                # Mark as visited
                self._visited_functions.add(call.callee_name)
                
                # Integrate the called SFM into the main SFM
                sfm = self._integrate_sfm(sfm, called_sfm, call)
                
                # Recursively expand calls in the integrated SFM
                if depth + 1 < self.max_depth:
                    sfm = self._expand_function_calls(sfm, depth + 1)
            
            except Exception as e:
                log_warning(f"Failed to expand {call.callee_name}: {e}")
        
        return sfm
    
    def _identify_function_calls(self, sfm: ScenarioFlowModel) -> list[FunctionCallSite]:
        """
        Identify function calls in the SFM that should be expanded.
        
        Looks for process nodes that represent function calls based on their labels.
        """
        calls = []
        
        for node in sfm.nodes:
            if node.node_type != "process":
                continue
            
            # Check if this is a function call that should be expanded
            # Look for patterns like "Validate X", "Process Y", "Handle Z"
            label = node.label.lower()
            
            # Extract potential function name from the label
            # Try to match common patterns
            func_name = self._extract_function_name_from_label(node.label)
            
            if func_name:
                calls.append(FunctionCallSite(
                    callee_name=func_name,
                    caller_context=node.label,
                    call_site_node_id=node.id,
                ))
        
        return calls
    
    def _extract_function_name_from_label(self, label: str) -> str | None:
        """
        Try to extract a function name from a node label.
        
        This is heuristic-based and looks for patterns that suggest a function call.
        """
        # Remove common prefixes
        label = label.strip()
        
        # Look for patterns like: "Call FunctionName", "FunctionName()", etc.
        patterns = [
            r"(?:Call|Invoke|Execute)\s+(\w+)",  # "Call FunctionName"
            r"(\w+)\s*\(",  # "FunctionName("
            r"^(\w+)$",  # Just "FunctionName"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, label, re.IGNORECASE)
            if match:
                func_name = match.group(1)
                # Check if it looks like a function name (PascalCase or camelCase)
                if func_name and (func_name[0].isupper() or "_" in func_name):
                    return func_name
        
        return None
    
    def _integrate_sfm(
        self,
        main_sfm: ScenarioFlowModel,
        called_sfm: ScenarioFlowModel,
        call_site: FunctionCallSite,
    ) -> ScenarioFlowModel:
        """
        Integrate a called function's SFM into the main SFM.
        
        Strategy:
        1. Find the call site node in the main SFM
        2. Replace it with the called function's SFM
        3. Reconnect edges appropriately
        """
        # For now, we'll append the called SFM's nodes as sub-nodes
        # A more sophisticated approach would be to truly inline them
        
        # Create a namespace for the called function's nodes
        prefix = f"{call_site.callee_name}_"
        
        # Copy called SFM nodes with prefixed IDs
        new_nodes = []
        id_map = {}
        
        for node in called_sfm.nodes:
            if node.id in {"start", "end"}:
                # Skip start/end from called function
                continue
            
            new_id = f"{prefix}{node.id}"
            id_map[node.id] = new_id
            
            new_nodes.append(SFMNode(
                id=new_id,
                node_type=node.node_type,
                label=f"{call_site.callee_name}: {node.label}",
                metadata=node.metadata,
            ))
        
        # Add new nodes to main SFM
        main_sfm.nodes.extend(new_nodes)
        
        # Copy edges with mapped IDs
        for edge in called_sfm.edges:
            if edge.src in id_map and edge.dst in id_map:
                from agent5.scenario_extractor import SFMEdge
                main_sfm.edges.append(SFMEdge(
                    src=id_map[edge.src],
                    dst=id_map[edge.dst],
                    label=edge.label,
                ))
        
        # Connect call site to the called function's flow
        if call_site.call_site_node_id and new_nodes:
            from agent5.scenario_extractor import SFMEdge
            
            # Find the first node in the called function
            first_node_id = new_nodes[0].id if new_nodes else None
            
            if first_node_id:
                # Redirect outgoing edges from call site to first node of called function
                for edge in main_sfm.edges:
                    if edge.src == call_site.call_site_node_id:
                        edge.dst = first_node_id
                        break
                else:
                    # No outgoing edge found, add one
                    main_sfm.edges.append(SFMEdge(
                        src=call_site.call_site_node_id,
                        dst=first_node_id,
                        label="call",
                    ))
        
        return main_sfm


def _set_parser_language(parser: Parser) -> None:
    """Set the C++ language for the tree-sitter parser."""
    raw = cpp_language()
    lang: Language
    if isinstance(raw, Language):
        lang = raw
    else:
        lang = Language(raw)  # type: ignore[arg-type]
    
    if hasattr(parser, "set_language"):
        parser.set_language(lang)  # type: ignore[attr-defined]
    else:
        parser.language = lang  # type: ignore[assignment]


def _node_text(source_bytes: bytes, node: Node) -> str:
    """Extract text from a tree-sitter node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def extract_cross_file_scenario(
    project_path: Path,
    entry_file: Path,
    entry_function: str,
    *,
    detail_level: DetailLevel = DetailLevel.MEDIUM,
    max_steps: int = 50,
    max_depth: int = 3,
) -> ScenarioFlowModel:
    """
    Extract a scenario flow that spans multiple files.
    
    Args:
        project_path: Root path of the C++ project
        entry_file: File containing the entry function
        entry_function: Name of the entry function
        detail_level: Level of detail for extraction
        max_steps: Maximum steps per function
        max_depth: Maximum recursion depth for following calls
        
    Returns:
        ScenarioFlowModel spanning multiple files
        
    Raises:
        RuntimeError: If entry function cannot be found or SFM cannot be built
    """
    extractor = CrossFileScenarioExtractor(
        project_path=project_path,
        detail_level=detail_level,
        max_steps=max_steps,
        max_depth=max_depth,
    )
    
    return extractor.extract_cross_file_scenario(entry_file, entry_function)





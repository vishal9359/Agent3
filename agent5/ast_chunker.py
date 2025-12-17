"""
AST-aware chunking for C++ code.

This module provides semantic chunking of C++ code based on the Abstract Syntax Tree (AST).
Instead of splitting code arbitrarily by character count, we chunk by semantic units
(functions, classes, namespaces) while preserving context and relationships.

Inspired by DocAgent's approach to understanding code structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter import Language, Node, Parser
from tree_sitter_cpp import language as cpp_language

from agent5.config import SETTINGS


@dataclass(frozen=True)
class CodeChunk:
    """A semantic chunk of code with metadata."""
    
    content: str
    chunk_type: str  # 'function', 'class', 'namespace', 'global', 'header'
    name: str  # Name of the entity (function/class/namespace)
    qualified_name: str  # Fully qualified name (e.g., 'Namespace::Class::method')
    start_line: int
    end_line: int
    file_path: str
    rel_path: str
    dependencies: list[str]  # Names of other entities this chunk depends on
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ChunkContext:
    """Context information for a chunk."""
    
    scope_stack: list[str]  # Current namespace/class scope
    file_includes: list[str]  # Include directives
    forward_decls: list[str]  # Forward declarations


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


def _get_node_name(source_bytes: bytes, node: Node) -> str | None:
    """Extract the name of a function/class/namespace from its node."""
    # Try the 'name' field first
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _node_text(source_bytes, name_node).strip()
    
    # For functions, try the 'declarator' field
    if node.type in {"function_definition", "constructor_or_destructor_definition"}:
        decl = node.child_by_field_name("declarator")
        if decl is not None:
            return _extract_identifier(source_bytes, decl)
    
    return None


def _extract_identifier(source_bytes: bytes, node: Node) -> str | None:
    """Extract an identifier from a node, handling various declarator types."""
    if node.type in {"identifier", "field_identifier", "type_identifier"}:
        return _node_text(source_bytes, node).strip()
    
    # Traverse to find identifier
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in {"identifier", "field_identifier", "type_identifier"}:
            text = _node_text(source_bytes, n).strip()
            if text:
                return text
        for c in reversed(n.children):
            stack.append(c)
    
    return None


def _extract_dependencies(source_bytes: bytes, node: Node) -> list[str]:
    """Extract function/class names that this node depends on."""
    deps: set[str] = set()
    
    # Find call expressions
    stack = [node]
    while stack:
        n = stack.pop()
        
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None:
                name = _extract_identifier(source_bytes, fn)
                if name:
                    deps.add(name)
        
        # Find type references
        elif n.type in {"type_identifier", "scoped_type_identifier"}:
            name = _node_text(source_bytes, n).strip()
            if name:
                deps.add(name)
        
        for c in n.children:
            stack.append(c)
    
    return sorted(deps)


def _extract_includes(source_bytes: bytes, root: Node) -> list[str]:
    """Extract all #include directives from the file."""
    includes: list[str] = []
    
    for child in root.children:
        if child.type == "preproc_include":
            text = _node_text(source_bytes, child).strip()
            includes.append(text)
    
    return includes


def _build_scope_name(scope_stack: list[str]) -> str:
    """Build a fully qualified name from the scope stack."""
    return "::".join(scope_stack) if scope_stack else ""


def _count_lines(text: str) -> int:
    """Count the number of lines in text."""
    return text.count("\n") + 1


class ASTChunker:
    """
    AST-aware code chunker for C++.
    
    Chunks code by semantic units (functions, classes, namespaces) while
    preserving context and relationships between entities.
    """
    
    def __init__(self):
        self.parser = Parser()
        _set_parser_language(self.parser)
    
    def chunk_file(
        self,
        file_path: Path,
        rel_path: str,
        *,
        min_lines: int | None = None,
        max_lines: int | None = None,
    ) -> list[CodeChunk]:
        """
        Chunk a C++ file into semantic units.
        
        Args:
            file_path: Absolute path to the file
            rel_path: Relative path from project root
            min_lines: Minimum lines per chunk (merge small chunks)
            max_lines: Maximum lines per chunk (split large chunks)
            
        Returns:
            List of CodeChunk objects
        """
        min_lines = min_lines or SETTINGS.min_chunk_lines
        max_lines = max_lines or SETTINGS.max_chunk_lines
        
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        
        if not source.strip():
            return []
        
        src_bytes = source.encode("utf-8", errors="ignore")
        tree = self.parser.parse(src_bytes)
        root = tree.root_node
        
        # Extract file-level context
        includes = _extract_includes(src_bytes, root)
        
        chunks: list[CodeChunk] = []
        scope_stack: list[str] = []
        
        # Track header content (includes, forward decls, macros)
        header_lines: list[str] = []
        header_end_line = 0
        
        for child in root.children:
            if child.type in {"preproc_include", "preproc_def", "preproc_ifdef", "comment"}:
                header_end_line = max(header_end_line, child.end_point[0])
        
        if header_end_line > 0:
            header_text = "\n".join(source.split("\n")[: header_end_line + 1])
            if header_text.strip():
                # Convert includes list to string for ChromaDB compatibility
                includes_str = "; ".join(includes) if includes else ""
                chunks.append(
                    CodeChunk(
                        content=header_text,
                        chunk_type="header",
                        name="<file header>",
                        qualified_name=f"{rel_path}::<header>",
                        start_line=1,
                        end_line=header_end_line + 1,
                        file_path=str(file_path),
                        rel_path=rel_path,
                        dependencies=[],
                        metadata={"includes": includes_str},
                    )
                )
        
        # Process the AST
        self._process_node(
            src_bytes,
            root,
            scope_stack,
            chunks,
            file_path,
            rel_path,
            includes,
            source,
            min_lines,
            max_lines,
        )
        
        return chunks
    
    def _process_node(
        self,
        src_bytes: bytes,
        node: Node,
        scope_stack: list[str],
        chunks: list[CodeChunk],
        file_path: Path,
        rel_path: str,
        includes: list[str],
        source: str,
        min_lines: int,
        max_lines: int,
    ) -> None:
        """Recursively process AST nodes to extract chunks."""
        
        # Handle scope-defining nodes
        if node.type in {"namespace_definition", "class_specifier", "struct_specifier"}:
            name = _get_node_name(src_bytes, node)
            
            if name:
                scope_stack.append(name)
                qualified_name = _build_scope_name(scope_stack)
                
                # Extract the entire scope as a chunk
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                content = _node_text(src_bytes, node)
                
                chunk_type = "namespace" if node.type == "namespace_definition" else "class"
                
                # Only create chunk if it's not too small
                if _count_lines(content) >= min_lines:
                    deps = _extract_dependencies(src_bytes, node)
                    
                    chunks.append(
                        CodeChunk(
                            content=content,
                            chunk_type=chunk_type,
                            name=name,
                            qualified_name=qualified_name,
                            start_line=start_line,
                            end_line=end_line,
                            file_path=str(file_path),
                            rel_path=rel_path,
                            dependencies=deps,
                            metadata={"scope": qualified_name},
                        )
                    )
            
            # Process children
            for child in node.children:
                self._process_node(
                    src_bytes,
                    child,
                    scope_stack,
                    chunks,
                    file_path,
                    rel_path,
                    includes,
                    source,
                    min_lines,
                    max_lines,
                )
            
            if name:
                scope_stack.pop()
        
        # Handle function definitions
        elif node.type in {"function_definition", "constructor_or_destructor_definition"}:
            name = _get_node_name(src_bytes, node)
            
            if name:
                qualified_name = _build_scope_name(scope_stack + [name])
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                content = _node_text(src_bytes, node)
                
                # Extract function signature separately for better context
                body = node.child_by_field_name("body")
                if body:
                    sig_end = body.start_byte
                    signature = src_bytes[node.start_byte:sig_end].decode("utf-8", errors="ignore").strip()
                else:
                    signature = content.split("{")[0].strip() if "{" in content else content
                
                deps = _extract_dependencies(src_bytes, node)
                
                chunks.append(
                    CodeChunk(
                        content=content,
                        chunk_type="function",
                        name=name,
                        qualified_name=qualified_name,
                        start_line=start_line,
                        end_line=end_line,
                        file_path=str(file_path),
                        rel_path=rel_path,
                        dependencies=deps,
                        metadata={
                            "scope": _build_scope_name(scope_stack),
                            "signature": signature,
                        },
                    )
                )
        
        # Recurse for other nodes (but don't go into function bodies we've already chunked)
        else:
            for child in node.children:
                self._process_node(
                    src_bytes,
                    child,
                    scope_stack,
                    chunks,
                    file_path,
                    rel_path,
                    includes,
                    source,
                    min_lines,
                    max_lines,
                )
    
    def merge_small_chunks(
        self,
        chunks: list[CodeChunk],
        min_lines: int | None = None,
    ) -> list[CodeChunk]:
        """
        Merge chunks that are too small to be useful on their own.
        
        Args:
            chunks: List of chunks to potentially merge
            min_lines: Minimum lines per chunk
            
        Returns:
            Merged list of chunks
        """
        min_lines = min_lines or SETTINGS.min_chunk_lines
        
        if not chunks:
            return []
        
        merged: list[CodeChunk] = []
        accumulator: list[CodeChunk] = []
        total_lines = 0
        
        for chunk in chunks:
            chunk_lines = chunk.end_line - chunk.start_line + 1
            
            # Keep header chunks separate
            if chunk.chunk_type == "header":
                if accumulator:
                    # Flush accumulator first
                    merged.extend(accumulator)
                    accumulator = []
                    total_lines = 0
                merged.append(chunk)
                continue
            
            # If chunk is large enough, flush accumulator and add it
            if chunk_lines >= min_lines:
                if accumulator:
                    merged.extend(accumulator)
                    accumulator = []
                    total_lines = 0
                merged.append(chunk)
            else:
                # Accumulate small chunks
                accumulator.append(chunk)
                total_lines += chunk_lines
                
                # If accumulated chunks are large enough, flush
                if total_lines >= min_lines:
                    merged.extend(accumulator)
                    accumulator = []
                    total_lines = 0
        
        # Flush remaining
        if accumulator:
            merged.extend(accumulator)
        
        return merged


def chunk_cpp_file(file_path: Path, rel_path: str) -> list[CodeChunk]:
    """
    Convenience function to chunk a C++ file.
    
    Args:
        file_path: Absolute path to the file
        rel_path: Relative path from project root
        
    Returns:
        List of semantic code chunks
    """
    chunker = ASTChunker()
    chunks = chunker.chunk_file(file_path, rel_path)
    return chunker.merge_small_chunks(chunks)


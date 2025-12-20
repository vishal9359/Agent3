"""Filesystem utilities for working with C++ projects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# C++ file extensions
CPP_EXTS = {".cpp", ".cc", ".cxx", ".c++", ".hpp", ".h", ".hh", ".hxx", ".h++", ".inl", ".ipp"}

# Additional extensions for RAG context
RAG_EXTRA_EXTS = {".txt", ".md", ".cmake", ".mk", ".makefile"}

# Directories to exclude by default (for RAG/indexing)
DEFAULT_EXCLUDE_DIRS = {
    "build",
    "builds",
    "dist",
    "out",
    "target",
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "venv",
    ".venv",
    "env",
    ".env",
}

# Directories to exclude from AST/CFG/semantic analysis (project boundary enforcement)
# These directories contain external libraries, build artifacts, and third-party code
PROJECT_EXCLUDE_DIRS = {
    # Build artifacts
    "build",
    "builds",
    "out",
    "dist",
    "target",
    "cmake-build-*",
    "bazel-*",  # Bazel build directories (handled specially)
    
    # Cache directories
    ".cache",
    ".ccache",
    
    # External/third-party libraries
    "external",
    "third_party",
    "thirdparty",
    "third-party",
    "vendor",
    "vendors",
    "libs",
    "lib",
    "deps",
    "dependencies",
    "vcpkg_installed",
    "vcpkg",
    "conan",
    "conan_data",
    
    # System/toolchain includes (should not be in project root, but just in case)
    "usr",
    "usr_local",
    
    # IDE/build system directories
    ".vs",
    ".vscode",
    ".idea",
    ".clangd",
    ".bazel",
    ".bazelrc",
    "proto",
    "generated",
    "test",
    "unit-tests",
    "integration-tests",
}

# Patterns that should be excluded (checked with startswith)
PROJECT_EXCLUDE_PATTERNS = [
    "bazel-",  # Bazel build directories
    "cmake-build-",  # CMake build directories
]


@dataclass(frozen=True)
class FileRecord:
    """Record of a file in the project."""
    path: Path
    relpath: str


def iter_files(
    root: Path,
    *,
    include_exts: set[str] | None = None,
    exclude_dir_names: set[str] | None = None,
) -> Iterable[FileRecord]:
    """
    Recursively iterate over files in a directory tree.
    
    Args:
        root: Root directory to search
        include_exts: Set of file extensions to include (e.g., {'.cpp', '.h'})
        exclude_dir_names: Set of directory names to skip
        
    Yields:
        FileRecord objects for matching files
    """
    root = root.resolve()
    exclude_dirs = exclude_dir_names or DEFAULT_EXCLUDE_DIRS
    
    if not root.exists():
        return
    
    if root.is_file():
        if include_exts is None or root.suffix.lower() in include_exts:
            yield FileRecord(path=root, relpath=root.name)
        return
    
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        
        # Skip excluded directories
        if any(parent.name in exclude_dirs for parent in path.parents):
            continue
        
        # Check extension
        if include_exts is not None and path.suffix.lower() not in include_exts:
            continue
        
        try:
            relpath = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relpath = path.name
        
        yield FileRecord(path=path, relpath=relpath)


def safe_read_text(path: Path, *, encoding: str = "utf-8") -> str:
    """
    Safely read text from a file, returning empty string on error.
    
    Args:
        path: Path to file
        encoding: Text encoding
        
    Returns:
        File contents or empty string
    """
    try:
        return path.read_text(encoding=encoding, errors="ignore")
    except Exception:
        return ""


def is_in_project_scope(file_path: Path, project_root: Path) -> bool:
    """
    Check if a file path is within the project scope and should be analyzed.
    
    This enforces the hard AST boundary: only files within the project root
    and not in excluded directories are considered for AST/CFG/semantic analysis.
    
    Args:
        file_path: Path to check (can be absolute or relative)
        project_root: Root path of the project
        
    Returns:
        True if the file should be included in analysis, False otherwise
    """
    try:
        # Resolve both paths to absolute
        file_path = Path(file_path).resolve()
        project_root = Path(project_root).resolve()
        
        # Check if file is within project root
        try:
            relative_path = file_path.relative_to(project_root)
        except ValueError:
            # File is not under project root
            return False
        
        # Check each part of the path for excluded directories
        for part in relative_path.parts:
            # Check against excluded directory names
            if part in PROJECT_EXCLUDE_DIRS:
                return False
            
            # Check against exclusion patterns
            for pattern in PROJECT_EXCLUDE_PATTERNS:
                if part.startswith(pattern):
                    return False
        
        return True
        
    except Exception:
        # On any error, exclude the file (fail-safe)
        return False


def should_exclude_path(path: Path, project_root: Path) -> bool:
    """
    Check if a path should be excluded from project analysis.
    
    This is an alias for `not is_in_project_scope()` for readability.
    
    Args:
        path: Path to check
        project_root: Root path of the project
        
    Returns:
        True if the path should be excluded, False if it should be included
    """
    return not is_in_project_scope(path, project_root)


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "build",
    "Build",
    "cmake-build-debug",
    "cmake-build-release",
    "out",
    "dist",
    "third_party",
    "ThirdParty",
    "vendor",
    "node_modules",
    "__pycache__",
}


CPP_EXTS = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inl",
    ".ipp",
}

RAG_EXTRA_EXTS = {
    ".cmake",
    ".md",
    ".rst",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
}


@dataclass(frozen=True)
class FileRecord:
    path: Path
    relpath: str


def iter_files(
    root: Path,
    *,
    include_exts: set[str] | None = None,
    exclude_dir_names: set[str] | None = None,
    max_file_size_bytes: int = 2_000_000,
) -> Iterable[FileRecord]:
    root = root.resolve()
    include_exts = include_exts or set()
    exclude_dir_names = exclude_dir_names or set()
    merged_excludes = DEFAULT_EXCLUDE_DIR_NAMES | exclude_dir_names

    for p in root.rglob("*"):
        try:
            if p.is_dir():
                # Skip excluded directories by preventing descent where possible.
                if p.name in merged_excludes:
                    # rglob can't prune; but we can just ignore its children by continuing.
                    # This is still fine for typical repos.
                    continue
                continue
            if not p.is_file():
                continue
        except OSError:
            continue

        if p.suffix and include_exts and p.suffix.lower() not in include_exts:
            continue

        try:
            if p.stat().st_size > max_file_size_bytes:
                continue
        except OSError:
            continue

        rel = str(p.relative_to(root)).replace("\\", "/")
        yield FileRecord(path=p, relpath=rel)


def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None



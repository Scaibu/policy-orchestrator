"""
================================================================================
BLUEPRINT: FILE SEARCH BUSINESS RULES
================================================================================
1. OVERVIEW:
   Evaluates business logic decision tables for file search filtering, path
   normalization, prune predicates, and algorithm selection validity.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Rule tables and evaluation criteria are defined in this top-level header.
   Functions remain 100% comment-free.
================================================================================
"""
from typing import Set


VALID_ALGORITHMS: Set[str] = {"DFS", "BFS"}
DEFAULT_PRUNE_NAMES: Set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
}


def is_valid_algorithm(algorithm: str) -> bool:
    return algorithm.upper() in VALID_ALGORITHMS


def should_prune_directory(dir_name: str, custom_exclude: Set[str]) -> bool:
    if dir_name in DEFAULT_PRUNE_NAMES:
        return True
    return dir_name in custom_exclude


def matches_pattern(filename: str, pattern: str) -> bool:
    if not pattern or pattern == "*":
        return True
    return filename.endswith(pattern) or pattern in filename

"""
================================================================================
BLUEPRINT: MODULE RESOLVER DECISION RULES (PEP 302 / PEP 451)
================================================================================
1. OVERVIEW:
   Implements module resolution rules matching Python's import machinery:
   - Priority Cascade: Iterates paths in exact sequential order.
   - Package Type Discrimination:
     * Regular Package: Directory containing '__init__.py'.
     * Single Module: File with '.py' extension.
     * Namespace Package: Directory without '__init__.py' (PEP 420).
   - Collision Detection: If multiple search paths contain the same module name,
     the first shadows subsequent paths, producing a collision alert.
================================================================================
"""
from pathlib import Path
from typing import List, Tuple, Optional

PKG_TYPE_REGULAR = "REGULAR_PACKAGE"
PKG_TYPE_MODULE = "MODULE_FILE"
PKG_TYPE_NAMESPACE = "NAMESPACE_PACKAGE"
PKG_TYPE_NOT_FOUND = "NOT_FOUND"


def inspect_search_path_entry(base_dir: Path, module_parts: List[str]) -> Tuple[Optional[Path], str]:
    target = base_dir.joinpath(*module_parts)

    file_py = target.with_suffix(".py")
    if file_py.is_file():
        return file_py, PKG_TYPE_MODULE

    init_py = target / "__init__.py"
    if init_py.is_file():
        return init_py, PKG_TYPE_REGULAR

    if target.is_dir():
        return target, PKG_TYPE_NAMESPACE

    return None, PKG_TYPE_NOT_FOUND


def resolve_module_cascade(module_name: str, search_paths: List[str]) -> Tuple[Optional[str], str, List[str], bool]:
    parts = module_name.strip().split(".")
    candidates: List[Tuple[Path, str]] = []

    for sp in search_paths:
        base = Path(sp).resolve()
        if not base.exists():
            continue
        resolved_path, pkg_type = inspect_search_path_entry(base, parts)
        if resolved_path is not None:
            candidates.append((resolved_path, pkg_type))

    if not candidates:
        return None, PKG_TYPE_NOT_FOUND, [], False

    winner_path, winner_type = candidates[0]
    shadowed = [str(p) for p, _ in candidates[1:]]
    has_collision = len(candidates) > 1

    return str(winner_path), winner_type, shadowed, has_collision

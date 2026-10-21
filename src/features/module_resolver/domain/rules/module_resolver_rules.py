"""
================================================================================
BLUEPRINT: MODULE RESOLVER DECISION RULES (PEP 302 / PEP 451 / PEP 566)
================================================================================
1. OVERVIEW:
   Implements complete Python module and package import resolution rules:
   - Builtin & Standard Library Identification: Detects C-builtins and sys.stdlib.
   - Search Path Inspection:
     * Regular Package: Directory containing '__init__.py'.
     * Single File Module: '.py' file.
     * Compiled Extensions: '.so', '.pyd', '.abi3.so' native binaries.
     * Namespace Package: Directory without '__init__.py' (PEP 420).
   - Collision & Shadowing Detection across sys.path hierarchy.
   - Metadata & Version Introspection: Resolves installed PyPI distribution and version.
   - Static AST Import Extractor: Scans Python source AST to discover all imports.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All rules adhere strictly to zero mid-function inline comments.
================================================================================
"""
import ast
import importlib.metadata
import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set

PKG_TYPE_BUILTIN = "BUILTIN_MODULE"
PKG_TYPE_STDLIB = "STDLIB_MODULE"
PKG_TYPE_COMPILED = "COMPILED_EXTENSION"
PKG_TYPE_REGULAR = "REGULAR_PACKAGE"
PKG_TYPE_MODULE = "MODULE_FILE"
PKG_TYPE_NAMESPACE = "NAMESPACE_PACKAGE"
PKG_TYPE_NOT_FOUND = "NOT_FOUND"

COMPILED_EXTENSIONS = (".so", ".pyd")


def get_stdlib_path() -> Optional[Path]:
    stdlib_dir = sysconfig.get_path("stdlib")
    return Path(stdlib_dir).resolve() if stdlib_dir else None


def is_builtin_module(module_name: str) -> bool:
    root_module = module_name.split(".")[0]
    return root_module in sys.builtin_module_names


def inspect_search_path_entry(base_dir: Path, module_parts: List[str]) -> Tuple[Optional[Path], str]:
    target = base_dir.joinpath(*module_parts)

    file_py = target.with_suffix(".py")
    if file_py.is_file():
        return file_py, PKG_TYPE_MODULE

    for ext in COMPILED_EXTENSIONS:
        ext_candidate = target.with_suffix(ext)
        if ext_candidate.is_file():
            return ext_candidate, PKG_TYPE_COMPILED
        glob_matches = list(target.parent.glob(f"{target.name}.*{ext}"))
        if glob_matches and glob_matches[0].is_file():
            return glob_matches[0], PKG_TYPE_COMPILED

    init_py = target / "__init__.py"
    if init_py.is_file():
        return init_py, PKG_TYPE_REGULAR

    for ext in COMPILED_EXTENSIONS:
        init_ext = target / f"__init__{ext}"
        if init_ext.is_file():
            return init_ext, PKG_TYPE_COMPILED

    if target.is_dir():
        return target, PKG_TYPE_NAMESPACE

    return None, PKG_TYPE_NOT_FOUND


def introspect_package_metadata(module_name: str, resolved_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    root_pkg = module_name.split(".")[0]
    try:
        dist_name = root_pkg
        version = importlib.metadata.version(root_pkg)
        return dist_name, version
    except importlib.metadata.PackageNotFoundError:
        pass

    try:
        packages_distributions = importlib.metadata.packages_distributions()
        dist_candidates = packages_distributions.get(root_pkg, [])
        if dist_candidates:
            dist_name = dist_candidates[0]
            version = importlib.metadata.version(dist_name)
            return dist_name, version
    except Exception:
        pass

    return None, None


def resolve_module_cascade(
    module_name: str, search_paths: List[str]
) -> Tuple[Optional[str], str, List[str], bool, Optional[str], Optional[str], bool, bool]:
    clean_name = module_name.strip()
    if is_builtin_module(clean_name):
        return None, PKG_TYPE_BUILTIN, [], False, None, sys.version.split()[0], True, True

    parts = clean_name.split(".")
    candidates: List[Tuple[Path, str]] = []

    for sp in search_paths:
        base = Path(sp).resolve()
        if not base.exists():
            continue
        resolved_path, pkg_type = inspect_search_path_entry(base, parts)
        if resolved_path is not None:
            candidates.append((resolved_path, pkg_type))

    if not candidates:
        try:
            spec = importlib.util.find_spec(clean_name)
            if spec is not None:
                origin = spec.origin
                if origin == "built-in":
                    return None, PKG_TYPE_BUILTIN, [], False, None, sys.version.split()[0], True, True
                if origin:
                    dist_name, version = introspect_package_metadata(clean_name, origin)
                    is_std = "lib/python" in origin and "site-packages" not in origin
                    return origin, PKG_TYPE_MODULE, [], False, dist_name, version, False, is_std
        except (ImportError, AttributeError, ValueError):
            pass
        return None, PKG_TYPE_NOT_FOUND, [], False, None, None, False, False

    winner_path, winner_type = candidates[0]
    shadowed = [str(p) for p, _ in candidates[1:]]
    has_collision = len(candidates) > 1

    dist_name, version = introspect_package_metadata(clean_name, str(winner_path))
    stdlib_path = get_stdlib_path()
    is_stdlib = stdlib_path is not None and str(stdlib_path) in str(winner_path) and "site-packages" not in str(winner_path)

    return str(winner_path), winner_type, shadowed, has_collision, dist_name, version, False, is_stdlib


def extract_ast_imports(file_path: Path) -> List[str]:
    if not file_path.is_file():
        return []
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except Exception:
        return []

    imported_modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    return sorted(list(imported_modules))


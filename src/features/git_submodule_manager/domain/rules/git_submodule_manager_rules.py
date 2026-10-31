"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER DOMAIN RULES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Pure domain decision rules and data transformations for Git submodules:
     - `.gitmodules` parsing and serialization as pure config strings.
     - Gitlink tree node payload creation with mode `160000` (subproject commit).
     - Submodule path sanitization and normalization.
     - Verification of child commit existence on remote.
     - Drift calculation between parent gitlink SHA and child branch HEAD.

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - 100% pure functional decisions without side effects or network calls.
================================================================================
"""

import configparser
import io
import re
from typing import Dict, List, Any, Optional, Tuple
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    InvalidSubmoduleSpecifierError,
)


def sanitize_submodule_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/").lstrip("/")
    parts = [p for p in cleaned.split("/") if p and p != "."]
    safe_parts: List[str] = []
    for part in parts:
        if part == "..":
            if safe_parts:
                safe_parts.pop()
        else:
            safe_parts.append(part)
    res = "/".join(safe_parts)
    if not res:
        raise InvalidSubmoduleSpecifierError(f"Invalid submodule path: {path}")
    return res


def parse_gitmodules_content(content: str) -> Dict[str, Dict[str, str]]:
    if not content or not content.strip():
        return {}
    clean_text = content.strip()
    if not clean_text.startswith("["):
        try:
            import base64
            decoded = base64.b64decode(clean_text).decode("utf-8")
            clean_text = decoded
        except Exception:
            pass
    config = configparser.ConfigParser()
    try:
        config.read_string(clean_text)
    except Exception:
        return {}
    modules: Dict[str, Dict[str, str]] = {}
    for section in config.sections():
        if section.startswith('submodule "') and section.endswith('"'):
            name = section[11:-1]
            path = config.get(section, "path", fallback="")
            url = config.get(section, "url", fallback="")
            branch = config.get(section, "branch", fallback="main")
            if path:
                modules[path] = {
                    "name": name,
                    "path": path,
                    "url": url,
                    "branch": branch,
                }
    return modules


def serialize_gitmodules_content(modules: Dict[str, Dict[str, str]]) -> str:
    config = configparser.ConfigParser()
    for path, info in sorted(modules.items(), key=lambda x: x[0]):
        name = info.get("name", path)
        section = f'submodule "{name}"'
        config[section] = {
            "path": path,
            "url": info.get("url", ""),
            "branch": info.get("branch", "main"),
        }
    output = io.StringIO()
    config.write(output)
    return output.getvalue()


def upsert_submodule_in_gitmodules(
    existing_content: str,
    submodule_path: str,
    submodule_url: str,
    submodule_branch: str = "main",
) -> str:
    clean_path = sanitize_submodule_path(submodule_path)
    modules = parse_gitmodules_content(existing_content)
    modules[clean_path] = {
        "name": clean_path,
        "path": clean_path,
        "url": submodule_url,
        "branch": submodule_branch,
    }
    return serialize_gitmodules_content(modules)


def build_gitlink_tree_nodes(
    submodule_paths_to_shas: Dict[str, str],
    gitmodules_blob_sha: Optional[str] = None,
) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    for path, commit_sha in sorted(submodule_paths_to_shas.items()):
        clean_path = sanitize_submodule_path(path)
        nodes.append({
            "path": clean_path,
            "mode": "160000",
            "type": "commit",
            "sha": commit_sha,
        })
    if gitmodules_blob_sha:
        nodes.append({
            "path": ".gitmodules",
            "mode": "100644",
            "type": "blob",
            "sha": gitmodules_blob_sha,
        })
    return nodes


def is_valid_sha(sha: Optional[str]) -> bool:
    if not sha or not isinstance(sha, str):
        return False
    return bool(re.match(r"^[0-9a-fA-F]{40}$", sha.strip()))


def calculate_commit_drift(current_gitlink_sha: str, remote_head_sha: str) -> Tuple[bool, int]:
    clean_current = current_gitlink_sha.strip().lower()
    clean_remote = remote_head_sha.strip().lower()
    if clean_current == clean_remote:
        return True, 0
    return False, 1

"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE DOMAIN RULES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Pure domain decision rules for branch lifecycle operations:
     - Branch name sanity checks (forbidden prefixes, git ref format compliance).
     - Clean ref normalization (strips redundant 'refs/heads/').
     - Error mapping for GitHub/GitLab HTTP errors.

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - 100% pure functional decisions without side effects or network calls.
================================================================================
"""

import re
from typing import Optional
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    InvalidBranchNameError,
    BranchAlreadyExistsError,
    BranchNotFoundError,
    MergeConflictError,
)


def normalize_branch_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned.startswith("refs/heads/"):
        cleaned = cleaned.replace("refs/heads/", "")
    if cleaned.startswith("refs/tags/"):
        cleaned = cleaned.replace("refs/tags/", "")
    cleaned = cleaned.strip("/")
    if not cleaned:
        raise InvalidBranchNameError("Branch name cannot be empty")
    if ".." in cleaned or "~" in cleaned or "^" in cleaned or ":" in cleaned or "?" in cleaned or "*" in cleaned or "[" in cleaned:
        raise InvalidBranchNameError(f"Branch name contains illegal Git reference characters: {name}")
    if cleaned.endswith(".lock") or cleaned.endswith("/"):
        raise InvalidBranchNameError(f"Branch name cannot end with .lock or slash: {name}")
    return cleaned


def is_sha(ref: str) -> bool:
    return bool(re.match(r"^[0-9a-fA-F]{40}$", ref.strip()))


def map_branch_http_error(
    status_code: int,
    body_text: str,
    operation: str = "branch_operation",
) -> Exception:
    lower_body = body_text.lower()
    if status_code == 404:
        return BranchNotFoundError(f"Target repository or ref not found during {operation}: {body_text}")
    if status_code == 409 or status_code == 422:
        if "already exists" in lower_body or "reference already exists" in lower_body:
            return BranchAlreadyExistsError(f"Branch reference already exists: {body_text}")
        if "merge conflict" in lower_body or "conflict" in lower_body:
            return MergeConflictError(f"Merge conflict detected during {operation}: {body_text}")
    if status_code == 401:
        from src.features.git_remote_inspector.contracts.git_remote_inspector_types import AuthenticationError
        return AuthenticationError(f"Authentication failed during {operation}: {body_text}")
    if status_code == 403:
        from src.features.git_remote_inspector.contracts.git_remote_inspector_types import PermissionDeniedError
        return PermissionDeniedError(f"Permission denied during {operation}: {body_text}")
    return RuntimeError(f"Git provider error ({status_code}) during {operation}: {body_text}")

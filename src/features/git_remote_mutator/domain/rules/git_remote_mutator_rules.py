import re
from typing import Optional, List, Dict, Any, Tuple
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    FileMutationEntry,
    StaleParentCommitError,
    InvalidRepositoryUrlError,
)


def validate_commit_sha(sha: Optional[str]) -> bool:
    if not sha or not isinstance(sha, str):
        return False
    return bool(re.match(r"^[0-9a-fA-F]{40}$", sha.strip()))


def sanitize_relative_path(path: str) -> str:
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
        raise ValueError(f"Invalid path: {path}")
    return res


def verify_optimistic_concurrency(current_head_sha: str, expected_parent_sha: Optional[str]) -> None:
    if expected_parent_sha is not None:
        clean_exp = expected_parent_sha.strip().lower()
        clean_curr = current_head_sha.strip().lower()
        if clean_exp != clean_curr:
            raise StaleParentCommitError(
                f"Optimistic concurrency check failed: expected parent SHA {clean_exp}, but remote branch is at {clean_curr}",
                details={"expected": clean_exp, "current": clean_curr},
            )


def calculate_git_blob_sha(content: str, encoding: str = "utf-8") -> str:
    import hashlib
    import base64
    if encoding == "base64":
        raw_bytes = base64.b64decode(content)
    else:
        raw_bytes = content.encode("utf-8")
    header = f"blob {len(raw_bytes)}\0".encode("ascii")
    return hashlib.sha1(header + raw_bytes).hexdigest()


def infer_git_file_mode(path: str, content: str = "", explicit_mode: Optional[str] = None) -> str:
    if explicit_mode and explicit_mode in ("100644", "100755", "120000", "160000"):
        return explicit_mode
    lower_path = path.lower()
    if lower_path.endswith((".sh", ".bash", ".zsh", ".command")):
        return "100755"
    if content.startswith("#!/") or content.startswith("#! /"):
        return "100755"
    return "100644"


def build_tree_nodes_payload(
    mutations: List[FileMutationEntry],
    blob_shas: Dict[str, str],
) -> List[Dict[str, Any]]:
    tree_nodes: List[Dict[str, Any]] = []
    for entry in mutations:
        clean_path = sanitize_relative_path(entry.path)
        resolved_mode = infer_git_file_mode(clean_path, entry.content, entry.mode)
        if entry.is_deleted:
            tree_nodes.append({
                "path": clean_path,
                "mode": resolved_mode,
                "type": "blob",
                "sha": None,
            })
        else:
            sha = blob_shas.get(clean_path)
            if sha:
                tree_nodes.append({
                    "path": clean_path,
                    "mode": resolved_mode,
                    "type": "blob",
                    "sha": sha,
                })
    return tree_nodes


def map_network_exception(exc: Exception) -> Exception:
    import requests
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        SSLError,
        DNSResolutionError,
        ConnectionTimeoutError,
        ReadTimeoutError,
    )
    if isinstance(exc, requests.exceptions.SSLError):
        return SSLError(str(exc))
    if isinstance(exc, requests.exceptions.ConnectionError):
        msg = str(exc)
        if "NameResolutionError" in msg or "getaddrinfo failed" in msg:
            return DNSResolutionError(msg)
        return ConnectionTimeoutError(msg)
    if isinstance(exc, requests.exceptions.Timeout):
        return ReadTimeoutError(str(exc))
    return exc


def map_mutation_http_error(
    status_code: int,
    body_text: str,
    provider: str,
    operation: str = "mutation",
) -> Exception:
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        AuthenticationError,
        PermissionDeniedError,
        RateLimitExceededError,
        ProviderServerError,
    )
    from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
        BranchNotFoundError,
        RefUpdateConflictError,
    )
    if status_code == 401:
        return AuthenticationError(f"{provider} authentication failed during {operation}: {body_text}")
    if status_code == 403:
        if "rate limit" in body_text.lower():
            return RateLimitExceededError(f"{provider} rate limit exceeded during {operation}: {body_text}")
        return PermissionDeniedError(f"{provider} write access forbidden for {operation}: {body_text}")
    if status_code == 404:
        return BranchNotFoundError(f"Target repository or ref not found during {operation}: {body_text}")
    if status_code in (409, 422):
        return RefUpdateConflictError(f"{provider} reference update conflict (fast-forward rejected or branch mutated concurrently): {body_text}")
    if status_code >= 500:
        return ProviderServerError(f"{provider} server error ({status_code}) during {operation}: {body_text}", status_code=status_code)
    return RuntimeError(f"{provider} error ({status_code}) during {operation}: {body_text}")

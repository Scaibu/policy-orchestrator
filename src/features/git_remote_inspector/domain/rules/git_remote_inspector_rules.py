import fnmatch
from typing import Optional, List, Dict, Any, Tuple
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import RemoteTreeItem


def sanitize_file_path(path: str) -> str:
    cleaned = path.strip().replace("\\", "/").lstrip("/")
    parts = [p for p in cleaned.split("/") if p and p != "."]
    safe_parts: List[str] = []
    for part in parts:
        if part == "..":
            if safe_parts:
                safe_parts.pop()
        else:
            safe_parts.append(part)
    return "/".join(safe_parts)


def matches_glob_pattern(path: str, pattern: Optional[str]) -> bool:
    if not pattern or pattern == "*" or pattern == "**":
        return True
    norm_path = path.lstrip("/")
    norm_pattern = pattern.lstrip("/")
    if fnmatch.fnmatch(norm_path, norm_pattern):
        return True
    if "/" in norm_pattern:
        return fnmatch.fnmatch(norm_path, f"*{norm_pattern}*")
    file_name = norm_path.split("/")[-1]
    return fnmatch.fnmatch(file_name, norm_pattern)


def filter_tree_nodes(
    tree_items: List[Dict[str, Any]], pattern: Optional[str]
) -> List[RemoteTreeItem]:
    results: List[RemoteTreeItem] = []
    for item in tree_items:
        path = str(item.get("path", ""))
        item_type = str(item.get("type", "blob"))
        if item_type == "tree":
            continue
        if matches_glob_pattern(path, pattern):
            results.append(
                RemoteTreeItem(
                    path=path,
                    mode=str(item.get("mode", "100644")),
                    item_type=item_type,
                    sha=str(item.get("sha", "")),
                    size=item.get("size"),
                )
            )
    return results


def build_nested_tree_hierarchy(
    raw_tree: List[Dict[str, Any]], max_depth: Optional[int] = None
) -> Tuple[Dict[str, Any], int, int]:
    root: Dict[str, Any] = {"type": "directory", "children": {}}
    file_count = 0
    dir_count = 0
    for item in raw_tree:
        path = str(item.get("path", "")).strip("/")
        if not path:
            continue
        item_type = str(item.get("type", "blob"))
        parts = path.split("/")
        if max_depth is not None and len(parts) > max_depth:
            continue
        current = root
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                if item_type == "tree":
                    if part not in current["children"]:
                        current["children"][part] = {"type": "directory", "children": {}}
                        dir_count += 1
                else:
                    current["children"][part] = {
                        "type": "file",
                        "sha": str(item.get("sha", "")),
                        "size": item.get("size"),
                        "mode": str(item.get("mode", "100644")),
                    }
                    file_count += 1
            else:
                if part not in current["children"]:
                    current["children"][part] = {"type": "directory", "children": {}}
                    dir_count += 1
                current = current["children"][part]
    return root.get("children", {}), file_count, dir_count


def extract_diff_patch(compare_payload: Dict[str, Any]) -> Tuple[int, str]:

    files = compare_payload.get("files", [])
    files_count = len(files)
    patches: List[str] = []
    for f in files:
        filename = f.get("filename", "")
        status = f.get("status", "modified")
        patch = f.get("patch", "")
        header = f"diff --git a/{filename} b/{filename}\nstatus: {status}\n"
        patches.append(f"{header}{patch}")
    return files_count, "\n\n".join(patches)


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


def map_http_status_error(
    status_code: int,
    body_text: str,
    provider: str,
    resource_type: str = "resource",
) -> Exception:
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        AuthenticationError,
        PermissionDeniedError,
        RateLimitExceededError,
        RefNotFoundError,
        FileNotFoundError,
        BlobNotFoundError,
        RepositoryNotFoundError,
        TreeTruncatedError,
        DiffTooLargeError,
        PayloadTooLargeError,
        ProviderServerError,
    )
    if status_code == 401:
        return AuthenticationError(f"{provider} authentication failed: {body_text}")
    if status_code == 403:
        if "rate limit" in body_text.lower():
            return RateLimitExceededError(f"{provider} rate limit exceeded: {body_text}")
        return PermissionDeniedError(f"{provider} access forbidden: {body_text}")
    if status_code == 404:
        if resource_type in ("tree", "ref"):
            return RefNotFoundError(f"Remote ref or tree not found: {body_text}")
        if resource_type == "file":
            return FileNotFoundError(f"Remote file not found: {body_text}")
        if resource_type == "blob":
            return BlobNotFoundError(f"Remote blob not found: {body_text}")
        return RepositoryNotFoundError(f"Remote repository or {resource_type} not found: {body_text}")
    if status_code == 422:
        if "tree" in body_text.lower():
            return TreeTruncatedError(f"{provider} tree error: {body_text}")
        if "diff" in body_text.lower():
            return DiffTooLargeError(f"{provider} diff too large: {body_text}")
        return PayloadTooLargeError(f"{provider} unprocessable entity: {body_text}")
    if status_code == 429:
        return RateLimitExceededError(f"{provider} rate limit exceeded: {body_text}")
    if status_code >= 500:
        return ProviderServerError(f"{provider} error ({status_code}): {body_text}", status_code=status_code)
    return RuntimeError(f"{provider} error ({status_code}): {body_text}")


LANGUAGE_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "python": (".py",),
    "typescript": (".ts", ".tsx"),
    "javascript": (".js", ".jsx", ".mjs"),
    "c": (".c", ".h"),
    "cpp": (".cpp", ".hpp", ".cc", ".cxx"),
    "java": (".java",),
    "go": (".go",),
    "rust": (".rs",),
    "kotlin": (".kt", ".kts"),
    "swift": (".swift",),
    "dart": (".dart",),
    "csharp": (".cs",),
}

CONTRACT_RULE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "category": "API_SPECIFICATION",
        "priority": 1,
        "keywords": ("openapi", "swagger"),
        "extensions": (".yaml", ".yml", ".json"),
    },
    {
        "category": "RPC_GRAPHQL_SCHEMA",
        "priority": 1,
        "keywords": (),
        "extensions": (".proto", ".graphql", ".gql"),
    },
    {
        "category": "DATA_CONTRACT_TYPES",
        "priority": 2,
        "keywords": ("types", "dto", "model", "contract", "schema"),
        "extensions": None,
    },
    {
        "category": "API_ROUTING_SURFACE",
        "priority": 3,
        "keywords": ("router", "routes", "controller", "endpoint"),
        "extensions": None,
    },
    {
        "category": "EVENT_MESSAGING_SCHEMA",
        "priority": 4,
        "keywords": ("event", "kafka", "consumer", "topic", "queue"),
        "extensions": None,
    },
    {
        "category": "DATABASE_MIGRATION",
        "priority": 5,
        "keywords": ("migration",),
        "extensions": None,
    },
]


def detect_file_language(lower_path: str) -> Optional[str]:
    for lang, exts in LANGUAGE_EXTENSIONS.items():
        if any(lower_path.endswith(ext) for ext in exts):
            return lang
    return None


def evaluate_contract_rule(lower_path: str, rule: Dict[str, Any], all_code_exts: Tuple[str, ...]) -> bool:
    keywords = rule["keywords"]
    extensions = rule["extensions"]
    if extensions is not None:
        if not any(lower_path.endswith(ext) for ext in extensions):
            return False
        if keywords and not any(kw in lower_path for kw in keywords):
            return False
        return True
    if not any(lower_path.endswith(ext) for ext in all_code_exts):
        return False
    if keywords and not any(kw in lower_path for kw in keywords):
        return False
    return True


def categorize_contract_nodes(raw_tree: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    discovered: List[Dict[str, Any]] = []
    lang_counts: Dict[str, int] = {lang: 0 for lang in LANGUAGE_EXTENSIONS}
    all_code_exts = tuple(ext for exts in LANGUAGE_EXTENSIONS.values() for ext in exts) + (".json", ".sql")

    for item in raw_tree:
        if item.get("type") != "blob":
            continue
        path = str(item.get("path", ""))
        lower_path = path.lower()
        sha = str(item.get("sha", ""))
        size = item.get("size")

        detected_lang = detect_file_language(lower_path)
        if detected_lang:
            lang_counts[detected_lang] += 1

        for rule in CONTRACT_RULE_DEFINITIONS:
            if evaluate_contract_rule(lower_path, rule, all_code_exts):
                discovered.append({
                    "category": rule["category"],
                    "priority": rule["priority"],
                    "path": path,
                    "sha": sha,
                    "size": size,
                })
                break

    primary_lang = max(lang_counts, key=lang_counts.get)
    if lang_counts[primary_lang] == 0:
        primary_lang = "generic"

    discovered.sort(key=lambda x: (x["priority"], x["path"]))
    return primary_lang, discovered


def match_entrypoint_file(raw_tree: List[Dict[str, Any]], target: str) -> Optional[str]:
    clean_target = target.strip("/").lower()
    candidates: List[str] = []
    for item in raw_tree:
        if item.get("type") != "blob":
            continue
        p = str(item.get("path", ""))
        lp = p.lower()
        if any(lp.endswith(ext) for ext in (".py", ".ts", ".js", ".go", ".java", ".rs")):
            if any(k in lp for k in ("router", "routes", "controller", "endpoint", "api")):
                candidates.append(p)

    for p in candidates:
        if clean_target in p.lower():
            return p

    if candidates:
        return candidates[0]
    return None


def extract_symbols_and_imports(content: str, file_path: str) -> Tuple[List[str], List[str], str]:
    invoked: List[str] = []
    imports: List[str] = []
    lower_fp = file_path.lower()

    if lower_fp.endswith(".py"):
        import ast
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        invoked.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        invoked.append(node.func.attr)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                    for n in node.names:
                        imports.append(n.name)
                elif isinstance(node, ast.Import):
                    for n in node.names:
                        imports.append(n.name)
        except Exception:
            pass

    if not invoked:
        import re
        call_matches = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", content)
        for m in call_matches:
            if m not in ("if", "for", "while", "return", "switch", "catch", "def", "function"):
                invoked.append(m)
        import_matches = re.findall(r"(?:import|from|require)\s+['\"]?([a-zA-Z0-9_./-]+)['\"]?", content)
        for im in import_matches:
            imports.append(im)

    unique_invoked = list(dict.fromkeys(invoked))[:10]
    unique_imports = list(dict.fromkeys(imports))[:10]

    layer = "UNKNOWN"
    if any(k in lower_fp for k in ("router", "controller", "routes", "endpoint")):
        layer = "ROUTER"
    elif any(k in lower_fp for k in ("service", "usecase", "domain", "workflow", "engine")):
        layer = "SERVICE"
    elif any(k in lower_fp for k in ("repo", "repository", "adapter", "client", "store", "db")):
        layer = "ADAPTER"
    elif any(k in lower_fp for k in ("models", "types", "dto", "schema", "entity")):
        layer = "DOMAIN_MODEL"

    return unique_invoked, unique_imports, layer


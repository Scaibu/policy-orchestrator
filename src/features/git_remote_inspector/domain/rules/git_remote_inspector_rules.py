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


def _resolve_schema_ref(schema: Any, components: Dict[str, Any], depth: int = 0) -> Any:
    if depth > 5 or not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref_path = str(schema["$ref"]).replace("#/components/schemas/", "").replace("#/components/", "")
        resolved = components.get(ref_path) or components.get(ref_path.split("/")[-1])
        if resolved and isinstance(resolved, dict):
            return _resolve_schema_ref(resolved, components, depth + 1)
        return schema
    resolved_schema: Dict[str, Any] = {}
    for k, v in schema.items():
        if k == "properties" and isinstance(v, dict):
            resolved_schema[k] = {prop_k: _resolve_schema_ref(prop_v, components, depth + 1) for prop_k, prop_v in v.items()}
        elif k == "items" and isinstance(v, dict):
            resolved_schema[k] = _resolve_schema_ref(v, components, depth + 1)
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            resolved_schema[k] = [_resolve_schema_ref(item, components, depth + 1) for item in v if item != {"type": "null"}]
        else:
            resolved_schema[k] = v
    return resolved_schema


def _generate_sample_payload(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return {}
    if "properties" in schema and isinstance(schema["properties"], dict):
        sample = {}
        for prop_name, prop_def in schema["properties"].items():
            if not isinstance(prop_def, dict):
                sample[prop_name] = "string"
                continue
            prop_type = prop_def.get("type", "string")
            default_val = prop_def.get("default")
            if default_val is not None:
                sample[prop_name] = default_val
            elif prop_type == "string":
                sample[prop_name] = prop_def.get("example", f"sample_{prop_name}")
            elif prop_type in ("integer", "number"):
                sample[prop_name] = prop_def.get("example", 1)
            elif prop_type == "boolean":
                sample[prop_name] = prop_def.get("example", True)
            elif prop_type == "array":
                item_schema = prop_def.get("items", {})
                sample[prop_name] = [_generate_sample_payload(item_schema)] if item_schema else []
            elif prop_type == "object":
                sample[prop_name] = _generate_sample_payload(prop_def)
            else:
                sample[prop_name] = "string"
        return sample
    return {}


def _generate_client_snippet(
    method: str, path: str, sample_body: Dict[str, Any], auth_scheme: Optional[str] = None
) -> Tuple[str, str]:
    import json
    json_body = json.dumps(sample_body, indent=2) if sample_body else ""
    curl_body = f"  -d '{json.dumps(sample_body)}' \\\n" if sample_body else ""

    auth_header_curl = ""
    auth_header_py = ""
    if auth_scheme == "BearerAuth" or auth_scheme == "OAuth2":
        auth_header_curl = "  -H 'Authorization: Bearer <TOKEN>' \\\n"
        auth_header_py = "        'Authorization': 'Bearer <TOKEN>',\n"
    elif auth_scheme == "ApiKeyAuth":
        auth_header_curl = "  -H 'X-API-KEY: <API_KEY>' \\\n"
        auth_header_py = "        'X-API-KEY': '<API_KEY>',\n"

    curl_cmd = (
        f"curl -X {method} 'http://<service-host>{path}' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -H 'traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' \\\n"
        f"  -H 'x-request-id: req-integration-sample' \\\n"
        f"{auth_header_curl}"
        f"{curl_body}"
        f"  | jq ."
    )

    code_snippet = (
        f"import requests\n\n"
        f"def call_{method.lower()}_{path.strip('/').replace('/', '_').replace('-', '_')}():\n"
        f"    url = f'http://{{HOST}}{path}'\n"
        f"    headers = {{\n"
        f"        'Content-Type': 'application/json',\n"
        f"        'traceparent': '00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01',\n"
        f"        'x-request-id': 'req-integ-client',\n"
        f"{auth_header_py}"
        f"    }}\n"
    )
    if sample_body:
        code_snippet += f"    payload = {json_body}\n"
        code_snippet += f"    response = requests.{method.lower()}(url, json=payload, headers=headers, timeout=10)\n"
    else:
        code_snippet += f"    response = requests.{method.lower()}(url, headers=headers, timeout=10)\n"
    code_snippet += f"    response.raise_for_status()\n    return response.json()\n"

    return curl_cmd, code_snippet



def extract_openapi_endpoint_signature(
    spec_text: str, target: str
) -> Optional[Dict[str, Any]]:
    import yaml
    try:
        data = yaml.safe_load(spec_text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    paths = data.get("paths", {})
    components = data.get("components", {}).get("schemas", {})
    if not isinstance(paths, dict):
        return None

    norm_target = target.strip().lower()
    matched_path = None
    matched_method = None
    matched_operation = None

    for path_str, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        clean_path = path_str.strip().lower()
        if clean_path == norm_target or norm_target in clean_path:
            for method in ("get", "post", "put", "patch", "delete", "options", "head"):
                op = path_item.get(method)
                if op and isinstance(op, dict):
                    matched_path = path_str
                    matched_method = method.upper()
                    matched_operation = op
                    break
        if matched_operation:
            break

    if not matched_operation:
        for path_str, path_item in paths.items():
            if not isinstance(path_item, dict):
                continue
            for method in ("get", "post", "put", "patch", "delete"):
                op = path_item.get(method)
                if isinstance(op, dict):
                    op_id = str(op.get("operationId", "")).lower()
                    summary = str(op.get("summary", "")).lower()
                    if norm_target in op_id or norm_target in summary:
                        matched_path = path_str
                        matched_method = method.upper()
                        matched_operation = op
                        break
            if matched_operation:
                break

    if not matched_operation or not matched_path:
        return None

    path_params: List[str] = []
    query_params: List[str] = []
    headers: List[str] = []

    for param in matched_operation.get("parameters", []):
        if not isinstance(param, dict):
            continue
        p_in = param.get("in")
        p_name = param.get("name", "")
        if p_in == "path":
            path_params.append(p_name)
        elif p_in == "query":
            query_params.append(p_name)
        elif p_in == "header":
            headers.append(p_name)

    body_schema: Dict[str, Any] = {}
    req_body = matched_operation.get("requestBody", {})
    if isinstance(req_body, dict):
        content = req_body.get("content", {})
        if isinstance(content, dict):
            app_json = content.get("application/json", {})
            if isinstance(app_json, dict) and "schema" in app_json:
                body_schema = _resolve_schema_ref(app_json["schema"], components)

    resp_schema: Dict[str, Any] = {}
    responses = matched_operation.get("responses", {})
    if isinstance(responses, dict):
        resp_200 = responses.get("200") or responses.get("201") or responses.get(200) or responses.get(201)
        if isinstance(resp_200, dict):
            content = resp_200.get("content", {})
            if isinstance(content, dict):
                app_json = content.get("application/json", {})
                if isinstance(app_json, dict) and "schema" in app_json:
                    resp_schema = _resolve_schema_ref(app_json["schema"], components)

    is_deprecated = bool(matched_operation.get("deprecated", False))
    auth_scheme = None
    security_reqs = matched_operation.get("security") or data.get("security", [])
    if isinstance(security_reqs, list) and security_reqs:
        first_sec = security_reqs[0]
        if isinstance(first_sec, dict) and first_sec:
            sec_name = list(first_sec.keys())[0]
            sec_def = data.get("components", {}).get("securitySchemes", {}).get(sec_name, {})
            sec_type = sec_def.get("type", "")
            sec_scheme = sec_def.get("scheme", "").lower()
            if sec_type == "http" and sec_scheme == "bearer":
                auth_scheme = "BearerAuth"
            elif sec_type == "apiKey":
                auth_scheme = "ApiKeyAuth"
            elif sec_type == "oauth2":
                auth_scheme = "OAuth2"
            else:
                auth_scheme = sec_name

    sample_payload = _generate_sample_payload(body_schema)
    curl_cmd, code_snippet = _generate_client_snippet(matched_method, matched_path, sample_payload, auth_scheme)

    return {
        "protocol": "HTTP_REST",
        "http_method": matched_method,
        "path": matched_path,
        "summary": matched_operation.get("summary"),
        "auth_scheme": auth_scheme,
        "is_deprecated": is_deprecated,
        "path_params": path_params,
        "query_params": query_params,
        "request_headers": headers,
        "request_body_schema": body_schema,
        "sample_request_payload": sample_payload,
        "response_200_schema": resp_schema,
        "sample_curl": curl_cmd,
        "client_code_snippet": code_snippet,
    }



def synthesize_endpoint_signature_from_chain(
    target: str,
    chain_nodes: List[Dict[str, Any]],
    code_blobs: Dict[str, str],
) -> Dict[str, Any]:
    import re
    http_method = "POST" if any(w in target.lower() for w in ("create", "add", "post", "save", "scan", "search")) else "GET"
    path = f"/{target.lstrip('/')}"
    headers = ["traceparent", "x-request-id", "x-correlation-id"]
    body_fields: Dict[str, Any] = {}

    for node in chain_nodes:
        fp = node.get("file_path", "")
        code = code_blobs.get(fp, "")
        if not code:
            continue
        pydantic_fields = re.findall(r"(\w+)\s*:\s*([A-Za-z0-9_\[\],\s]+)(?:\s*=\s*Field\(.*?\))?", code)
        for field_name, field_type in pydantic_fields:
            if field_name not in ("model_config", "self", "cls"):
                clean_type = field_type.split("=")[0].strip()
                body_fields[field_name] = clean_type

        ts_fields = re.findall(r"(\w+)\s*(\??)\s*:\s*([A-Za-z0-9_<>,\[\]\s]+);?", code)
        for f_name, opt, f_type in ts_fields:
            if f_name not in body_fields and f_name not in ("constructor", "return", "if", "for"):
                body_fields[f_name] = f_type.strip()

        go_fields = re.findall(r"(\w+)\s+([A-Za-z0-9_\[\]*]+)\s+`json:\"([a-zA-Z0-9_-]+)\"", code)
        for _, g_type, j_name in go_fields:
            body_fields[j_name] = g_type

        java_fields = re.findall(r"private\s+([A-Za-z0-9_<>]+)\s+(\w+);", code)
        for j_type, j_name in java_fields:
            body_fields[j_name] = j_type

        proto_fields = re.findall(r"\s*(?:optional|repeated)?\s*([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*=\s*\d+;", code)
        for p_type, p_name in proto_fields:
            body_fields[p_name] = p_type

    body_schema = {
        "type": "object",
        "properties": {k: {"type": v} for k, v in list(body_fields.items())[:10]},
    } if body_fields else {}

    auth_scheme = None
    for code in code_blobs.values():
        if any(kw in code for kw in ("JwtAuthGuard", "get_current_user", "BearerAuth", "Authorize")):
            auth_scheme = "BearerAuth"
            break
        elif any(kw in code for kw in ("ApiKeyAuth", "x-api-key", "X-API-KEY")):
            auth_scheme = "ApiKeyAuth"
            break

    protocol = "GRPC" if any(fp.endswith(".proto") for fp in code_blobs.keys()) else "HTTP_REST"
    sample_payload = {k: f"sample_{k}" if any(t in str(v).lower() for t in ("str", "string")) else 1 if any(t in str(v).lower() for t in ("int", "num", "float", "double")) else True for k, v in body_fields.items()}
    curl_cmd, code_snippet = _generate_client_snippet(http_method, path, sample_payload, auth_scheme)

    return {
        "protocol": protocol,
        "http_method": http_method,
        "path": path,
        "summary": f"Synthesized integration signature for {target} ({protocol})",
        "auth_scheme": auth_scheme,
        "is_deprecated": False,
        "path_params": [],
        "query_params": [],
        "request_headers": headers,
        "request_body_schema": body_schema,
        "sample_request_payload": sample_payload,
        "response_200_schema": {"type": "object", "properties": {"success": {"type": "boolean"}, "data": {"type": "object"}}},
        "sample_curl": curl_cmd,
        "client_code_snippet": code_snippet,
    }




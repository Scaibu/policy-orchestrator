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

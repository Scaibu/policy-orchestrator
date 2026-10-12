from typing import List, Dict, Any, Tuple

def paginate_list(items: List[Any], page: int = 1, page_size: int = 20) -> Tuple[List[Any], Dict[str, Any]]:
    safe_page = max(1, page)
    safe_size = max(1, page_size)
    total_items = len(items)
    total_pages = (total_items + safe_size - 1) // safe_size if total_items > 0 else 0
    start_idx = (safe_page - 1) * safe_size
    end_idx = start_idx + safe_size
    sliced = items[start_idx:end_idx]

    pagination_meta = {
        "page": safe_page,
        "pageSize": safe_size,
        "totalItems": total_items,
        "totalPages": total_pages,
        "hasNext": safe_page < total_pages,
        "hasPrev": safe_page > 1,
    }
    return sliced, pagination_meta

def sanitize_json_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            sanitized[k] = sanitize_json_dict(v)
        elif isinstance(v, (list, tuple)):
            sanitized[k] = [sanitize_json_dict(i) if isinstance(i, dict) else i for i in v]
        elif v is not None:
            sanitized[k] = v
    return sanitized

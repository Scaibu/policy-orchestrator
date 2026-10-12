import hashlib
import uuid
from typing import Union

def hash_sha256(data: Union[str, bytes]) -> str:
    if isinstance(data, str):
        raw = data.encode("utf-8")
    else:
        raw = data
    return hashlib.sha256(raw).hexdigest()

def generate_uuid() -> str:
    return str(uuid.uuid4())

def mask_token(token: str, visible_chars: int = 4) -> str:
    if len(token) <= visible_chars * 2:
        return "*" * len(token)
    return token[:visible_chars] + "*" * (len(token) - (visible_chars * 2)) + token[-visible_chars:]

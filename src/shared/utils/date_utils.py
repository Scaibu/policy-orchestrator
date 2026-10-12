import time
from datetime import datetime, timezone
from typing import Optional

def format_iso_timestamp(dt: Optional[datetime] = None) -> str:
    target_dt = dt or datetime.now(timezone.utc)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    return target_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

def parse_iso_timestamp(ts_str: str) -> datetime:
    cleaned = ts_str.rstrip("Z")
    if "." in cleaned:
        dt = datetime.fromisoformat(cleaned)
    else:
        dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def parse_duration_seconds(duration_str: str) -> int:
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])
    if unit == "s":
        return value
    elif unit == "m":
        return value * 60
    elif unit == "h":
        return value * 3600
    elif unit == "d":
        return value * 86400
    return int(duration_str)

def monotonic_time_ms() -> int:
    return int(time.monotonic() * 1000)

def compute_elapsed_ms(start_time_ms: int) -> int:
    return max(0, monotonic_time_ms() - start_time_ms)

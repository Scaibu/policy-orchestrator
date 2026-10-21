"""
================================================================================
BLUEPRINT: CODE SEARCH REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound repository port defining persistence boundary for code pattern
   search telemetry, match counts, and duration metrics.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class CodeSearchRepositoryPort(ABC):
    @abstractmethod
    def record_search(self, record_id: str, pattern_count: int, matches_found: int, duration_ms: float) -> None:
        pass

    @abstractmethod
    def get_search_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        pass

"""
================================================================================
BLUEPRINT: FILE SEARCH REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound repository port defining the abstract persistence boundary for file
   search execution records and traversal metrics.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Interface contracts are defined in this top blueprint.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class FileSearchRepositoryPort(ABC):
    @abstractmethod
    def record_search(self, record_id: str, root_path: str, total_found: int, duration_ms: float) -> None:
        pass

    @abstractmethod
    def get_search_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        pass

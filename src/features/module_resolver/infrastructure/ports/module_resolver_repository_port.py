"""
================================================================================
BLUEPRINT: MODULE RESOLVER REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound repository port defining the persistence boundary for module
   resolution audits, collision detections, and execution traces.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class ModuleResolverRepositoryPort(ABC):
    @abstractmethod
    def record_resolution(
        self, record_id: str, module_name: str, is_valid: bool, has_collision: bool, duration_ms: float
    ) -> None:
        pass

    @abstractmethod
    def get_resolution_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        pass

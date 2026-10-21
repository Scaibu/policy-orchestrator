"""
================================================================================
BLUEPRINT: GIT VERIFIER REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound repository port defining the persistence boundary for Git
   verification records, submodule audit outcomes, and branch status.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class GitVerifierRepositoryPort(ABC):
    @abstractmethod
    def record_verification(
        self, record_id: str, repo_path: str, is_clean: bool, submodules_count: int, duration_ms: float
    ) -> None:
        pass

    @abstractmethod
    def get_verification_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        pass

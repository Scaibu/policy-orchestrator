"""
================================================================================
BLUEPRINT: GIT REMOTE INSPECTOR REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound repository port defining the persistence boundary for Git remote
   inspection ledger records, Merkle tree relationship graphs, and endpoint cache.
2. ARCHITECTURAL LOCATION:
   src/features/git_remote_inspector/infrastructure/ports/git_remote_inspector_repository_port.py
3. INVARIANTS:
   - Zero inline comments inside method signatures.
   - Pure abstract methods returning Kernel Result types.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitRemoteInspectorRepositoryPort(ABC):
    @abstractmethod
    def save_inspection_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        pass

    @abstractmethod
    def get_inspection_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        pass

    def find_cached_endpoint(
        self,
        ctx: TraceContext,
        repo_url: str,
        commit_sha: str,
        endpoint_or_entity: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        return Result.ok(None)

    def save_endpoint_relationship(
        self,
        ctx: TraceContext,
        repo_url: str,
        commit_sha: str,
        ref_name: str,
        result_data: Dict[str, Any],
    ) -> Result[bool, Exception]:
        return Result.ok(True)

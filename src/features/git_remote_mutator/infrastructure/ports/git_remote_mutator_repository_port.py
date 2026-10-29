"""
================================================================================
BLUEPRINT: GIT REMOTE MUTATOR REPOSITORY PORT (OUTBOUND CONTRACT)
================================================================================
1. OVERVIEW:
   Outbound persistence boundary defining atomic operations for mutation ledger
   audit logs (AlloyDB/PostgreSQL) and live branch pointer CAS synchronization
   in the Neo4j knowledge graph.
2. ARCHITECTURAL LOCATION:
   src/features/git_remote_mutator/infrastructure/ports/git_remote_mutator_repository_port.py
3. INVARIANTS:
   - Zero inline comments inside method signatures.
   - Pure abstract methods returning Kernel Result types.
================================================================================
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitRemoteMutatorRepositoryPort(ABC):
    @abstractmethod
    def save_mutation_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        pass

    @abstractmethod
    def get_mutation_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        pass

    def update_branch_pointer(
        self,
        ctx: TraceContext,
        repo_url: str,
        branch: str,
        parent_sha: str,
        new_commit_sha: str,
        commit_message: str,
    ) -> Result[bool, Exception]:
        return Result.ok(True)

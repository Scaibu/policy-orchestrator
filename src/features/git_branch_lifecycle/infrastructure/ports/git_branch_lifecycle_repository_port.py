"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE DRIVEN REPOSITORY PORT
================================================================================
1. OVERVIEW & OBJECTIVE:
   Outbound driven persistence port contract for branch lifecycle records:
     - Persist lifecycle ledger record (AlloyDB / PostgreSQL / SQLite).
     - Synchronize branch graph topology (Neo4j).

2. INVARIANTS:
   - ZERO comments inside method bodies.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitBranchLifecycleRepositoryPort(ABC):
    @abstractmethod
    def save_lifecycle_record(
        self, ctx: TraceContext, record: Dict[str, Any]
    ) -> Result[bool, Exception]:
        raise NotImplementedError

    @abstractmethod
    def create_branch_node(
        self, ctx: TraceContext, repo: str, branch: str, commit_sha: str
    ) -> Result[bool, Exception]:
        raise NotImplementedError

    @abstractmethod
    def delete_branch_node(
        self, ctx: TraceContext, repo: str, branch: str
    ) -> Result[bool, Exception]:
        raise NotImplementedError

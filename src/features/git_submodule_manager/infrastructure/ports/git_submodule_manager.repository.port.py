"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER REPOSITORY PORT (OUTBOUND PERSISTENCE CONTRACT)
================================================================================
1. OVERVIEW & OBJECTIVE:
   Outbound persistence boundary defining atomic operations for:
     - Submodule audit ledger persistence (AlloyDB / PostgreSQL / SQLite).
     - Live knowledge graph synchronization (:GitSubmodule, :SUBMODULE_OF, :POINTS_TO) in Neo4j.

2. INVARIANTS:
   - ZERO comments inside method signatures or bodies.
   - Pure abstract methods returning Kernel Result types.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitSubmoduleManagerRepositoryPort(ABC):
    @abstractmethod
    def save_submodule_record(
        self, ctx: TraceContext, record: Dict[str, Any]
    ) -> Result[bool, Exception]:
        raise NotImplementedError

    @abstractmethod
    def get_submodule_record(
        self, ctx: TraceContext, record_id: str
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        raise NotImplementedError

    @abstractmethod
    def sync_submodule_graph(
        self,
        ctx: TraceContext,
        parent_repo: str,
        submodule_path: str,
        submodule_url: str,
        gitlink_sha: str,
    ) -> Result[bool, Exception]:
        raise NotImplementedError

"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE DRIVEN CLIENT PORT
================================================================================
1. OVERVIEW & OBJECTIVE:
   Outbound driven port contract for remote Git provider operations:
     - Create branch ref (`POST /git/refs`).
     - Delete branch ref (`DELETE /git/refs/heads/{branch}`).
     - Merge branch (`POST /merges`).
     - Create repository (`POST /user/repos` or `POST /orgs/{org}/repos`).
     - Resolve ref / get branch HEAD (`GET /git/ref/heads/{branch}`).

2. INVARIANTS:
   - ZERO comments inside method bodies.
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result


class GitBranchLifecycleClientPort(ABC):
    @abstractmethod
    def resolve_ref_sha(
        self, ctx: TraceContext, repo: str, ref: str, token: Optional[str] = None
    ) -> Result[str, Exception]:
        raise NotImplementedError

    @abstractmethod
    def create_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, target_sha: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        raise NotImplementedError

    @abstractmethod
    def delete_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, token: Optional[str] = None
    ) -> Result[bool, Exception]:
        raise NotImplementedError

    @abstractmethod
    def merge_branches(
        self,
        ctx: TraceContext,
        repo: str,
        base: str,
        head: str,
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        raise NotImplementedError

    @abstractmethod
    def create_remote_repo(
        self,
        ctx: TraceContext,
        name: str,
        org: Optional[str] = None,
        description: Optional[str] = "",
        is_private: bool = True,
        auto_init: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        raise NotImplementedError

"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: TASK LIFECYCLE DATA MODELS & DTOS
================================================================================
Defines pure dataclass/pydantic data models, input command envelopes, and results
for automated Git and GitHub task lifecycle orchestration.

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

from typing import Optional, List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StartTaskCommand:
    repo: str
    title: str
    slug: str
    base_branch: str = "main"
    body: Optional[str] = None


@dataclass(frozen=True)
class StartTaskResult:
    repo: str
    issue_number: int
    issue_url: str
    branch_name: str
    status: str = "SUCCESS"


@dataclass(frozen=True)
class FinishTaskCommand:
    repo: str
    issue_number: int
    pr_title: Optional[str] = None
    pr_body: Optional[str] = None
    base_branch: str = "main"
    test_cmd: Optional[str] = None
    sync_parent_pointer: bool = True


@dataclass(frozen=True)
class FinishTaskResult:
    repo: str
    issue_number: int
    pr_number: int
    pr_url: str
    merged: bool
    branch_deleted: bool
    parent_synced: bool
    status: str = "SUCCESS"


@dataclass(frozen=True)
class SyncPointerCommand:
    submodule_path: str
    parent_repo_path: str = "."
    commit_msg: Optional[str] = None


@dataclass(frozen=True)
class SyncPointerResult:
    submodule_path: str
    commit_hash: str
    parent_commit: str
    status: str = "SUCCESS"


"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER CONTRACT TYPES & DTOS
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone and multi-repo Git submodule domain models and DTOs:
     - Add submodule: updates .gitmodules, fetches child HEAD, stages 160000 gitlink commit.
     - Update / pull submodule: updates submodule gitlink pointer to target commit/branch.
     - Push multi-repo: verifies child submodules commits exist before pushing parent commit.
     - Submodule sync status: audits and detects drift between parent gitlink and child branch HEAD.
     - Strongly typed domain exceptions adhering to RFC envelope status standards.

2. INVARIANTS:
   - ZERO comments inside class bodies (declared in this header docblock).
   - Strict Pydantic model configurations (extra='forbid').
================================================================================
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.shared.errors.application_error import ApplicationError


class AddSubmoduleCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target parent repository URL (e.g. owner/repo)")
    branch: str = Field(default="main", description="Target branch on parent repository to mutate")
    submodule_path: str = Field(..., description="Relative path in parent repository for the submodule")
    submodule_url: str = Field(..., description="Remote Git URL for the child submodule repository")
    submodule_ref: str = Field(default="main", description="Target branch or commit SHA of the child submodule")
    commit_message: str = Field(default="chore: add submodule", description="Git commit message for the submodule addition")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class UpdateSubmoduleCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target parent repository URL")
    branch: str = Field(default="main", description="Target branch on parent repository to mutate")
    submodule_path: str = Field(..., description="Relative path in parent repository for the submodule")
    target_commit_sha: Optional[str] = Field(default=None, description="Optional explicit target commit SHA to pin gitlink to")
    submodule_branch: Optional[str] = Field(default=None, description="Optional submodule branch name to pull latest HEAD from")
    commit_message: str = Field(default="chore: update submodule pointer", description="Git commit message for pointer update")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class PullSubmodulesCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target parent repository URL")
    branch: str = Field(default="main", description="Target branch on parent repository")
    submodule_paths: Optional[List[str]] = Field(default=None, description="Optional subset of submodule paths to update; if None, updates all")
    commit_message: str = Field(default="chore: pull and sync submodules", description="Git commit message for batch update")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class PushMultiRepoCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parent_repo_url: str = Field(..., description="Target parent repository URL")
    parent_branch: str = Field(default="main", description="Target parent branch name")
    submodule_paths: List[str] = Field(..., description="List of submodule paths whose commits must precede parent push")
    verify_remote_exists: bool = Field(default=True, description="Verify child commits are already pushed to remote remotes before updating parent")
    commit_message: str = Field(default="chore: synchronize multi-repo submodules", description="Parent commit message")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for OCC")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class SubmoduleSyncStatusCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target parent repository URL")
    branch: str = Field(default="main", description="Target branch on parent repository")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class SubmoduleItemStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    url: str
    branch: str
    gitlink_sha: str
    remote_head_sha: str
    is_in_sync: bool
    drift_commit_count: int = 0


class SubmoduleOperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    branch: str
    action: Literal["ADD_SUBMODULE", "UPDATE_SUBMODULE", "PULL_SUBMODULES", "PUSH_MULTI_REPO", "SYNC_STATUS"]
    parent_commit_sha: str
    new_commit_sha: str
    tree_sha: str
    submodules_affected: List[str] = Field(default_factory=list)
    commit_message: str
    is_noop: bool = False
    duration_ms: float = 0.0
    status_details: Optional[List[SubmoduleItemStatus]] = None


class GitSubmoduleBaseError(ApplicationError):
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        canonical_code: str = "INTERNAL_SERVER_ERROR",
        retryable: bool = False,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            canonical_code=canonical_code,
            retryable=retryable,
            details=details,
        )


class SubmoduleAlreadyExistsError(GitSubmoduleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=False,
            details=details,
        )


class SubmoduleNotFoundError(GitSubmoduleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            canonical_code="RESOURCE_NOT_FOUND",
            retryable=False,
            details=details,
        )


class SubmoduleRemoteCommitMissingError(GitSubmoduleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            canonical_code="UNPROCESSABLE_ENTITY",
            retryable=False,
            details=details,
        )


class InvalidSubmoduleSpecifierError(GitSubmoduleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            canonical_code="VALIDATION_FAILED",
            retryable=False,
            details=details,
        )

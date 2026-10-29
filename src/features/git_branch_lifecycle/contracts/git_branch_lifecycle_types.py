"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE CONTRACT TYPES & DTOS
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone Git branch lifecycle domain models and DTOs:
     - Branch creation from SHA or base ref.
     - Safe branch deletion.
     - Automated and fast-forward branch merges.
     - Remote repository initialization.
     - Typed domain exceptions adhering to RFC envelope status standards.

2. INVARIANTS:
   - ZERO comments inside class bodies (declared in this header docblock).
   - Strict Pydantic model configurations (extra='forbid').
================================================================================
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.shared.errors.application_error import ApplicationError


class CreateBranchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL (e.g. owner/repo)")
    branch_name: str = Field(..., description="Name of the new branch to create (without refs/heads/)")
    source_ref: str = Field(..., description="Source commit SHA or source branch name to fork from")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class DeleteBranchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    branch_name: str = Field(..., description="Name of the branch to delete")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class MergeBranchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    base_branch: str = Field(..., description="The name of the base branch that the head will be merged into")
    head_branch: str = Field(..., description="The head branch to merge")
    commit_message: Optional[str] = Field(default=None, description="Commit message to use for the merge commit")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class CreateRepositoryCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_name: str = Field(..., description="Name of the repository to create")
    org: Optional[str] = Field(default=None, description="Optional organization name. If omitted, created for authenticated user")
    description: Optional[str] = Field(default="", description="Repository description")
    is_private: bool = Field(default=True, description="Whether the repository should be private")
    auto_init: bool = Field(default=True, description="Initialize with an empty README")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class BranchLifecycleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    action: Literal["CREATE_BRANCH", "MERGE_BRANCH", "DELETE_BRANCH", "CREATE_REPO"]
    branch_name: Optional[str] = None
    source_ref: Optional[str] = None
    target_ref: Optional[str] = None
    result_sha: Optional[str] = None
    is_success: bool = True
    execution_ms: float = 0.0
    details: Optional[Dict[str, Any]] = None


class GitBranchLifecycleBaseError(ApplicationError):
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


class BranchAlreadyExistsError(GitBranchLifecycleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=False,
            details=details,
        )


class BranchNotFoundError(GitBranchLifecycleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            canonical_code="RESOURCE_NOT_FOUND",
            retryable=False,
            details=details,
        )


class MergeConflictError(GitBranchLifecycleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=False,
            details=details,
        )


class InvalidBranchNameError(GitBranchLifecycleBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            canonical_code="VALIDATION_FAILED",
            retryable=False,
            details=details,
        )

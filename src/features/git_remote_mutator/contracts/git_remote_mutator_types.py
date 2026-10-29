from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from src.shared.errors.application_error import ApplicationError


class FileMutationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="Target repository relative file path")
    content: str = Field(default="", description="UTF-8 or raw string content for the file")
    encoding: str = Field(default="utf-8", description="Encoding format: utf-8 or base64")
    mode: Optional[str] = Field(default=None, description="Git file mode (e.g. 100644 for blob, 100755 for executable)")
    is_deleted: bool = Field(default=False, description="Whether this entry marks a file deletion")


class MutateFileCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL (e.g. owner/repo or https://github.com/owner/repo)")
    branch: str = Field(default="main", description="Target branch name to mutate")
    file_path: str = Field(..., description="Relative file path to create, update, or overwrite")
    content: str = Field(..., description="New file content string")
    encoding: str = Field(default="utf-8", description="Encoding format: utf-8 or base64")
    mode: Optional[str] = Field(default=None, description="Optional Git file mode override")
    commit_message: str = Field(..., description="Authoritative Git commit message")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class MutateBatchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    branch: str = Field(default="main", description="Target branch name to mutate")
    mutations: List[FileMutationEntry] = Field(..., description="List of file mutation entries to apply atomically")
    commit_message: str = Field(..., description="Git commit message for the batch transaction")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class DeleteFileCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    branch: str = Field(default="main", description="Target branch name")
    file_path: str = Field(..., description="Relative file path to delete from remote branch")
    commit_message: str = Field(..., description="Git commit message for deletion")
    expected_parent_sha: Optional[str] = Field(default=None, description="Optional parent commit SHA for Optimistic Concurrency Control")
    author_name: Optional[str] = Field(default=None, description="Optional commit author name")
    author_email: Optional[str] = Field(default=None, description="Optional commit author email")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class RemoteBlobCreatedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha: str
    size: int


class RemoteMutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    branch: str
    parent_commit_sha: str
    new_commit_sha: str
    tree_sha: str
    files_mutated_count: int
    commit_message: str
    is_noop: bool = False
    blobs_created: List[RemoteBlobCreatedResult] = Field(default_factory=list)
    duration_ms: float = 0.0


class GitRemoteMutatorBaseError(ApplicationError):
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


class StaleParentCommitError(GitRemoteMutatorBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=False,
            details=details,
        )


class RefUpdateConflictError(GitRemoteMutatorBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=True,
            details=details,
        )


class BranchNotFoundError(GitRemoteMutatorBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            canonical_code="RESOURCE_NOT_FOUND",
            retryable=False,
            details=details,
        )


class InvalidRepositoryUrlError(GitRemoteMutatorBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            canonical_code="VALIDATION_FAILED",
            retryable=False,
            details=details,
        )


class EmptyMutationBatchError(GitRemoteMutatorBaseError):
    def __init__(self, message: str = "Mutation batch must contain at least one file entry", details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            canonical_code="VALIDATION_FAILED",
            retryable=False,
            details=details,
        )


class DatabaseConnectionError(GitRemoteMutatorBaseError):
    def __init__(self, message: str = "Database connection failed", details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            canonical_code="INTERNAL_SERVER_ERROR",
            retryable=True,
            details=details,
        )


class DatabaseConstraintViolationError(GitRemoteMutatorBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=409,
            canonical_code="CONFLICT",
            retryable=False,
            details=details,
        )


class DatabaseTimeoutError(GitRemoteMutatorBaseError):
    def __init__(self, message: str = "Database operation timed out", details: Optional[Any] = None) -> None:
        super().__init__(
            message=message,
            status_code=504,
            canonical_code="TIMEOUT",
            retryable=True,
            details=details,
        )

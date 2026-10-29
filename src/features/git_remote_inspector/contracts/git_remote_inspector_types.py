from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RemoteTreeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    mode: str = "100644"
    item_type: str = "blob"
    sha: str
    size: Optional[int] = None


class SearchFilesCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL (e.g. owner/repo or https://github.com/owner/repo)")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    path_pattern: Optional[str] = Field(default=None, description="Optional glob filter (e.g. *.ts, src/**)")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class GetFileContentCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    file_path: str = Field(..., description="Target relative file path")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class GetDiffCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    base_ref: str = Field(..., description="Base branch or commit SHA")
    head_ref: str = Field(..., description="Head branch or commit SHA")
    path: Optional[str] = Field(default=None, description="Optional specific file path filter")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class GetTreeHierarchyCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL (e.g. owner/repo or https://github.com/owner/repo)")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    max_depth: Optional[int] = Field(default=None, description="Optional maximum directory depth")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class RemoteFolderHierarchyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    commit_sha: str
    total_files: int
    total_directories: int
    truncated: bool = False
    tree_hierarchy: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class RemoteBranchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    commit_sha: str
    is_default: bool = False
    protected: bool = False


class ListBranchesCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL (e.g. owner/repo or https://github.com/owner/repo)")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class RemoteBranchesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    default_branch: str
    total_branches: int
    branches: List[RemoteBranchItem] = Field(default_factory=list)
    duration_ms: float = 0.0


class DiscoverContractsCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class DiscoveredContractItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    priority: int
    path: str
    sha: str
    size: Optional[int] = None


class RemoteDiscoveredContractsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    commit_sha: str
    primary_language: str
    total_contracts: int
    contracts: List[DiscoveredContractItem] = Field(default_factory=list)
    duration_ms: float = 0.0


class TraceCallChainCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    endpoint_or_symbol: str = Field(..., description="Target API endpoint (e.g. /login or /audit/scan) or function symbol name")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    max_depth: int = Field(default=3, description="Maximum call stack trace depth (1-5)")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class CallChainNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    depth: int
    layer: str
    file_path: str
    symbol_name: str
    invoked_symbols: List[str] = Field(default_factory=list)


class RemoteCallChainResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    target_endpoint_or_symbol: str
    entrypoint_file: Optional[str] = None
    call_chain: List[CallChainNode] = Field(default_factory=list)
    total_depth_reached: int = 0
    duration_ms: float = 0.0
class IntegrateContextCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str = Field(..., description="Target repository URL")
    endpoint_or_entity: str = Field(..., description="Target API endpoint (e.g. /login) or entity/symbol name")
    ref_name: str = Field(default="main", description="Branch name or commit SHA")
    subpath: Optional[str] = Field(default=None, description="Optional monorepo subfolder path filter (e.g. services/auth)")
    token: Optional[str] = Field(default=None, description="Optional per-request override access token")


class ExtractedEndpointSignature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: str = "HTTP_REST"
    http_method: str = "POST"
    path: str = ""
    summary: Optional[str] = None
    auth_scheme: Optional[str] = None
    is_deprecated: bool = False
    path_params: List[str] = Field(default_factory=list)
    query_params: List[str] = Field(default_factory=list)
    request_headers: List[str] = Field(default_factory=list)
    request_body_schema: Dict[str, Any] = Field(default_factory=dict)
    sample_request_payload: Dict[str, Any] = Field(default_factory=dict)
    response_200_schema: Dict[str, Any] = Field(default_factory=dict)
    sample_curl: str = ""
    client_code_snippet: str = ""


class RemoteIntegrateContextResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    target_endpoint_or_entity: str
    match_type: str = "OPENAPI_SPEC"
    contract_file: Optional[str] = None
    signature: ExtractedEndpointSignature
    related_dto_files: List[str] = Field(default_factory=list)
    call_chain: List[CallChainNode] = Field(default_factory=list)
    duration_ms: float = 0.0



class RemoteInspectionResult(BaseModel):


    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    commit_sha: str
    total_found: int
    truncated: bool = False
    files: List[RemoteTreeItem] = Field(default_factory=list)
    duration_ms: float = 0.0


class RemoteFileContentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    ref_name: str
    file_path: str
    blob_sha: str
    content: str
    encoding: str = "utf-8"
    size_bytes: int = 0
    duration_ms: float = 0.0


class RemoteDiffResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    provider: str = "github"
    base_ref: str
    head_ref: str
    files_changed_count: int = 0
    diff_patch: str = ""
    duration_ms: float = 0.0



class GitRemoteBaseError(Exception):
    def __init__(self, message: str, status_code: int = 500, canonical_code: str = "INTERNAL_SERVER_ERROR", retryable: bool = False, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.status_code: int = status_code
        self.canonical_code: str = canonical_code
        self.retryable: bool = retryable
        self.details: Optional[Any] = details


class InvalidRepositoryUrlError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=400, canonical_code="VALIDATION_FAILED", retryable=False, details=details)


class AuthenticationError(GitRemoteBaseError):
    def __init__(self, message: str = "Invalid credentials or token expired", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=401, canonical_code="UNAUTHORIZED", retryable=False, details=details)


class PermissionDeniedError(GitRemoteBaseError):
    def __init__(self, message: str = "Insufficient permissions to access repository", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=403, canonical_code="FORBIDDEN", retryable=False, details=details)


class RepositoryNotFoundError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=404, canonical_code="RESOURCE_NOT_FOUND", retryable=False, details=details)


class RefNotFoundError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=404, canonical_code="RESOURCE_NOT_FOUND", retryable=False, details=details)


class FileNotFoundError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=404, canonical_code="RESOURCE_NOT_FOUND", retryable=False, details=details)


class BlobNotFoundError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=404, canonical_code="RESOURCE_NOT_FOUND", retryable=False, details=details)


class RateLimitExceededError(GitRemoteBaseError):
    def __init__(self, message: str = "Remote Git API rate limit exceeded", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=429, canonical_code="RATE_LIMIT_EXCEEDED", retryable=True, details=details)


class PayloadTooLargeError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=413, canonical_code="PAYLOAD_TOO_LARGE", retryable=False, details=details)


class TreeTruncatedError(GitRemoteBaseError):
    def __init__(self, message: str = "Repository tree exceeded provider limit and was truncated", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=422, canonical_code="TREE_TRUNCATED", retryable=False, details=details)


class DiffTooLargeError(GitRemoteBaseError):
    def __init__(self, message: str = "Diff comparison is too large to render", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=422, canonical_code="DIFF_TOO_LARGE", retryable=False, details=details)


class ConnectionTimeoutError(GitRemoteBaseError):
    def __init__(self, message: str = "Connection to remote Git provider timed out", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=504, canonical_code="TIMEOUT", retryable=True, details=details)


class ReadTimeoutError(GitRemoteBaseError):
    def __init__(self, message: str = "Read timeout waiting for remote Git provider response", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=504, canonical_code="TIMEOUT", retryable=True, details=details)


class DNSResolutionError(GitRemoteBaseError):
    def __init__(self, message: str = "Could not resolve Git host DNS", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=502, canonical_code="DNS_RESOLUTION_FAILED", retryable=True, details=details)


class SSLError(GitRemoteBaseError):
    def __init__(self, message: str = "SSL certificate verification failed for remote Git host", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=502, canonical_code="SSL_VERIFICATION_FAILED", retryable=False, details=details)


class ProviderServerError(GitRemoteBaseError):
    def __init__(self, message: str, status_code: int = 502, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=status_code, canonical_code="INTERNAL_SERVER_ERROR", retryable=True, details=details)


class UnsupportedGitProviderError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=400, canonical_code="VALIDATION_FAILED", retryable=False, details=details)


class DatabaseConnectionError(GitRemoteBaseError):
    def __init__(self, message: str = "Database connection failed", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=500, canonical_code="INTERNAL_SERVER_ERROR", retryable=True, details=details)


class DatabaseConstraintViolationError(GitRemoteBaseError):
    def __init__(self, message: str, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=409, canonical_code="CONFLICT", retryable=False, details=details)


class DatabaseTimeoutError(GitRemoteBaseError):
    def __init__(self, message: str = "Database operation timed out", details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=504, canonical_code="TIMEOUT", retryable=True, details=details)


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


class RemoteInspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
    ref_name: str
    commit_sha: str
    total_found: int
    files: List[RemoteTreeItem] = Field(default_factory=list)
    duration_ms: float = 0.0


class RemoteFileContentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_url: str
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
    base_ref: str
    head_ref: str
    files_changed_count: int = 0
    diff_patch: str = ""
    duration_ms: float = 0.0

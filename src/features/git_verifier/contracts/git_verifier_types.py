"""
================================================================================
BLUEPRINT: GIT VERIFIER DOMAIN TYPES & DTO CONTRACTS
================================================================================
1. OVERVIEW:
   Domain models, commands, and results for Git Merkle tree state, submodule
   gitlink verification (mode 160000), and branch status audits.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All field specifications, default values, and schema invariants are defined here.
================================================================================
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SubmoduleStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str
    expected_commit: Optional[str] = None
    actual_commit: Optional[str] = None
    is_synchronized: bool = False
    is_dirty: bool = False


class WorkingTreeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    staged_files: List[str] = Field(default_factory=list)
    unstaged_files: List[str] = Field(default_factory=list)
    untracked_files: List[str] = Field(default_factory=list)
    is_clean: bool = True


class RemoteTrackingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upstream_branch: Optional[str] = None
    ahead_count: int = 0
    behind_count: int = 0
    is_in_sync: bool = True


class VerifyGitCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_path: str = Field(..., description="Root path of git repository to verify")
    check_submodules: bool = Field(default=True, description="Whether to verify submodule commit pointers")
    check_working_tree: bool = Field(default=True, description="Whether to audit staged and unstaged files")


class GitVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_path: str
    current_branch: Optional[str] = None
    head_commit: Optional[str] = None
    is_detached: bool = False
    is_clean: bool = True
    remote: RemoteTrackingStatus = Field(default_factory=RemoteTrackingStatus)
    working_tree: WorkingTreeStatus = Field(default_factory=WorkingTreeStatus)
    submodules: List[SubmoduleStatus] = Field(default_factory=list)
    has_submodule_mismatch: bool = False
    duration_ms: float = 0.0
    status: str = "COMPLETED"


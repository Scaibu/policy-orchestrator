"""
================================================================================
BLUEPRINT: FILE SEARCH DOMAIN MODELS & DTO CONTRACTS
================================================================================
1. OVERVIEW:
   Declares immutable Pydantic schemas and DTOs for the file search domain.
   Includes flat file listings as well as full recursive tree node hierarchies
   (directory -> children nodes with type, size, depth, and child arrays).
2. ZERO-INLINE-COMMENT DOCTRINE:
   All field specifications, default values, and schema invariants are defined here.
   Class bodies remain strictly comment-free.
================================================================================
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class FileSearchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_path: str = Field(..., description="Root directory path to start traversal")
    pattern: Optional[str] = Field(default=None, description="Filename match pattern or extension")
    algorithm: str = Field(default="DFS", description="Traversal algorithm: DFS or BFS")
    max_depth: int = Field(default=10, ge=1, le=50, description="Maximum directory traversal depth")
    exclude_dirs: List[str] = Field(
        default_factory=lambda: [".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"],
        description="Directory names to prune",
    )
    include_dirs: bool = Field(default=True, description="Whether to include directory nodes in the response")


class FileNodeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    name: str
    size_bytes: int
    is_dir: bool
    depth: int


class TreeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    name: str
    is_dir: bool
    size_bytes: int
    depth: int
    children: List["TreeNode"] = Field(default_factory=list)


TreeNode.model_rebuild()


class FileSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_path: str
    total_found: int
    total_dirs: int = 0
    total_files: int = 0
    files: List[FileNodeItem]
    tree: Optional[TreeNode] = None
    duration_ms: float
    status: str

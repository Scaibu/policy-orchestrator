from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


class CreateFileCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="Target file path")
    role: str = Field(..., description="Architectural role")
    rel_type: str = Field(..., description="Relationship type")
    target: str = Field(..., description="Target node ID or file path")
    direction: str = Field(default="outgoing", description="Relationship direction: outgoing | incoming")
    feature_name: Optional[str] = Field(default=None, description="Feature name")
    package_name: Optional[str] = Field(default=None, description="Package name")
    content: Optional[str] = Field(default=None, description="Optional initial file content")


class LinkFilesCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(..., description="Source node ID or file path")
    rel_type: str = Field(..., description="Relationship type")
    target: str = Field(..., description="Target node ID or file path")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Relationship metadata")


class ScaffoldFeatureCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    feature_name: str = Field(..., description="Name of the feature")
    base_dir: str = Field(default="src/features", description="Base directory")
    package_root: Optional[str] = Field(default=None, description="Optional parent package root")


class ScaffoldPackageCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    package_name: str = Field(..., description="Name of the package")
    base_dir: str = Field(default=".", description="Base directory")


class ScaffoldedFileNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    file_path: str
    role: str
    layer: str
    feature_name: Optional[str] = None
    package_name: Optional[str] = None
    size_bytes: int = 0


class ScaffoldedRelationshipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    source_id: str
    target_id: str
    rel_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class CreateFileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node: ScaffoldedFileNode
    relationship: ScaffoldedRelationshipRecord
    created_on_disk: bool
    status: str


class LinkFilesResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relationship: ScaffoldedRelationshipRecord
    status: str


class ScaffoldExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_name: str
    target_path: str
    total_directories: int
    total_files: int
    status: str

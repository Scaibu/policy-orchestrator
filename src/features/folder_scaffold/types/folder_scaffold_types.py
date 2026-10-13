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


class ScaffoldTreeNodeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="Target file path")
    role: str = Field(..., description="Architectural role")
    rel_type: str = Field(..., description="Initial relationship type to parent/target")
    target: str = Field(..., description="Target node ID or path")
    direction: str = Field(default="outgoing", description="Relationship direction: outgoing | incoming")
    feature_name: Optional[str] = Field(default=None, description="Feature name")
    package_name: Optional[str] = Field(default=None, description="Package name")
    content: Optional[str] = Field(default=None, description="Optional initial content")


class ScaffoldTreeRelationshipSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(..., description="Source node ID or path")
    rel_type: str = Field(..., description="Relationship type")
    target: str = Field(..., description="Target node ID or path")
    properties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata")


class ScaffoldTreeCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_id: str = Field(..., description="Root package/feature identifier or path")
    root_role: str = Field(default="PackageRoot", description="Role of the root node")
    nodes: List[ScaffoldTreeNodeSpec] = Field(default_factory=list, description="Ordered list of nodes to create")
    relationships: List[ScaffoldTreeRelationshipSpec] = Field(
        default_factory=list, description="Cross-node relationships to link"
    )
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Arbitrary run metadata")


class ScaffoldTreeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root_id: str
    total_created_files: int
    total_relationships: int
    nodes: List[ScaffoldedFileNode]
    relationships: List[ScaffoldedRelationshipRecord]
    status: str


class ImpactNodeDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    file_path: str
    role: str
    layer: str
    depth: int
    relationship: str
    risk_factor: str = "LOW"


class BlastRadiusReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    direction: str
    max_depth: int
    risk_level: str
    blast_radius_score: float
    impacted_nodes: List[ImpactNodeDetail]
    total_impacted: int
    execution_time_ms: int = 0


class FileLineageReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    file_path: str
    role: str
    layer: str
    direct_upstream: List[Dict[str, Any]]
    direct_downstream: List[Dict[str, Any]]
    blast_radius_score: float
    is_root_ingress: bool
    is_leaf_node: bool

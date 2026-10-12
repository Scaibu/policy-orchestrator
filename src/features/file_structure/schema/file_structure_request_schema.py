from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

def to_camel(string: str) -> str:
    components = string.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

class BaseApiSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )

class ScaffoldFeatureRequestSchema(BaseApiSchema):
    feature_name: str = Field(..., min_length=1, max_length=128, description="Feature name to scaffold in snake_case or kebab-case")
    base_dir: str = Field(default="src/features", description="Base features directory")
    package_root: Optional[str] = Field(default=None, description="Optional root path of the parent package (e.g. '.' or 'packages/order-service')")
    with_router: bool = Field(default=True, description="Whether to generate dedicated REST router in src/api/rest/v1/routers/")
    with_handler: bool = Field(default=True, description="Whether to generate dedicated REST handler in src/api/rest/v1/handlers/")
    port_type: str = Field(default="local", pattern="^(local|shared)$", description="Whether to scaffold a local repository port or link shared domain port")

class ScaffoldPackageRequestSchema(BaseApiSchema):
    package_name: str = Field(..., min_length=1, max_length=128, description="Package name to scaffold")
    base_dir: str = Field(default=".", description="Base directory to place package")

class ValidateStructureRequestSchema(BaseApiSchema):
    target_path: str = Field(..., min_length=1, description="Target feature or package path to validate against architectural rules")
    structure_type: str = Field(default="feature", pattern="^(feature|package)$", description="Type of structure to validate")

class AnalyzeImpactRequestSchema(BaseApiSchema):
    target_id: str = Field(..., min_length=1, max_length=256, description="Target node ID or file path")
    max_depth: int = Field(default=6, ge=1, le=15, description="Maximum graph search depth")
    direction: str = Field(default="UPSTREAM", pattern="^(UPSTREAM|DOWNSTREAM)$", description="Search direction")

class ScanRepositoryRequestSchema(BaseApiSchema):
    root_dir: str = Field(..., min_length=1, description="Root directory to scan and index")

class LinkFilesRequestSchema(BaseApiSchema):
    source: str = Field(..., min_length=1, description="Source file path or node ID")
    rel_type: str = Field(..., min_length=1, description="Relationship type (e.g. IMPORTS_SHARED, INVOKES_DOMAIN, CALLS_SERVICE, etc.)")
    target: str = Field(..., min_length=1, description="Target file path or node ID")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional relationship properties")

class CreateFileRequestSchema(BaseApiSchema):
    file_path: str = Field(..., min_length=1, description="File path to create")
    role: str = Field(..., min_length=1, description="Architectural role / label (e.g. DomainService, RuleSet, etc.)")
    rel_type: str = Field(..., min_length=1, description="Relationship type to target")
    target_file_or_node: str = Field(..., min_length=1, description="Target file path or node ID")
    direction: str = Field(default="outgoing", pattern="^(outgoing|incoming)$", description="Relationship direction")
    content: Optional[str] = Field(default=None, description="Optional file content template")
    package_root: Optional[str] = Field(default=None, description="Optional package root")
    feature_name: Optional[str] = Field(default=None, description="Optional feature name")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional node properties")

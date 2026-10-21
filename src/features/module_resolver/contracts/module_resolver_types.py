"""
================================================================================
BLUEPRINT: MODULE RESOLVER DOMAIN TYPES & DTO CONTRACTS
================================================================================
1. OVERVIEW:
   Domain models, commands, and results for Python module import resolution.
   Detects resolved file paths, package types (regular vs namespace), shadowing,
   and collision conditions.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All field specifications, default values, and schema invariants are defined here.
================================================================================
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ResolveModuleCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module_name: str = Field(..., description="Dotted module path (e.g. 'src.features.file_search')")
    search_paths: List[str] = Field(..., min_length=1, description="List of root directory paths simulating sys.path")


class ModuleResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module_name: str
    is_valid: bool
    resolved_path: Optional[str] = None
    package_type: str = "UNKNOWN"
    shadowed_paths: List[str] = Field(default_factory=list)
    has_collision: bool = False
    duration_ms: float = 0.0
    status: str = "COMPLETED"

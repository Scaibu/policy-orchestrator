"""
================================================================================
BLUEPRINT: CODE SEARCH DOMAIN TYPES & DTO CONTRACTS
================================================================================
1. OVERVIEW:
   Domain models, commands, and results for pattern matching across code files.
   Supports single-pattern Boyer-Moore-Horspool and multi-pattern Aho-Corasick.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All field specifications, default values, and schema invariants are defined here.
================================================================================
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class CodeSearchMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str
    line_number: int
    char_offset: int
    pattern: str
    line_content: str


class CodeSearchCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns: List[str] = Field(..., min_length=1, description="List of literal patterns to search for")
    target_files: List[str] = Field(..., min_length=1, description="List of file paths to search within")
    algorithm: str = Field(default="AHO_CORASICK", description="Algorithm: AHO_CORASICK or BOYER_MOORE")
    case_sensitive: bool = Field(default=True, description="Whether search is case-sensitive")
    max_matches_per_file: int = Field(default=100, ge=1, le=1000, description="Max matches per file")


class CodeSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_matches: int
    scanned_files_count: int
    matches: List[CodeSearchMatch]
    duration_ms: float
    status: str

"""
================================================================================
BLUEPRINT: FILE SEARCH ACL SCHEMA MAPPER
================================================================================
1. OVERVIEW:
   Provides bidirectional Anti-Corruption Layer (ACL) transformations between API
   payloads and domain types for file search operations.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Payload validation and serialization mappings are documented in this header.
   Function bodies remain strictly comment-free.
================================================================================
"""
from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.file_search.contracts.file_search_types import (
    FileSearchCommand,
    FileSearchResult,
)


def from_api_search_files(payload: Dict[str, Any]) -> Result[FileSearchCommand, ValidationError]:
    try:
        cmd = FileSearchCommand.model_validate(payload)
        return Result.ok(cmd)
    except PydanticValidationError as exc:
        details = [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "issue": err["msg"],
                "rule": err["type"],
            }
            for err in exc.errors()
        ]
        return Result.err(ValidationError(message="Validation failed for file search payload", details=details))


def to_api_search_files(result: FileSearchResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class FileSearchSchema:
    from_api_search_files = staticmethod(from_api_search_files)
    to_api_search_files = staticmethod(to_api_search_files)

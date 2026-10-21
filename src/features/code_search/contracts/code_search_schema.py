"""
================================================================================
BLUEPRINT: CODE SEARCH ACL SCHEMA MAPPER
================================================================================
1. OVERVIEW:
   Provides bidirectional Anti-Corruption Layer (ACL) transformations between API
   payloads and domain types for code pattern search.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Payload validation and serialization mappings are documented in this header.
================================================================================
"""
from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.code_search.contracts.code_search_types import (
    CodeSearchCommand,
    CodeSearchResult,
)


def from_api_search_code(payload: Dict[str, Any]) -> Result[CodeSearchCommand, ValidationError]:
    try:
        cmd = CodeSearchCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for code search payload", details=details))


def to_api_search_code(result: CodeSearchResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class CodeSearchSchema:
    from_api_search_code = staticmethod(from_api_search_code)
    to_api_search_code = staticmethod(to_api_search_code)

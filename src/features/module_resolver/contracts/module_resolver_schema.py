"""
================================================================================
BLUEPRINT: MODULE RESOLVER ACL SCHEMA MAPPER
================================================================================
1. OVERVIEW:
   Provides bidirectional Anti-Corruption Layer (ACL) transformations between API
   payloads and domain types for Python module import resolution.
================================================================================
"""
from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.module_resolver.contracts.module_resolver_types import (
    ResolveModuleCommand,
    ModuleResolutionResult,
)


def from_api_resolve_module(payload: Dict[str, Any]) -> Result[ResolveModuleCommand, ValidationError]:
    try:
        cmd = ResolveModuleCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for module resolve payload", details=details))


def to_api_resolve_module(result: ModuleResolutionResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class ModuleResolverSchema:
    from_api_resolve_module = staticmethod(from_api_resolve_module)
    to_api_resolve_module = staticmethod(to_api_resolve_module)

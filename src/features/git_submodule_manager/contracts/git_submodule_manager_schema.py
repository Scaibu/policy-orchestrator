"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER SCHEMA & ANTI-CORRUPTION LAYER (ACL)
================================================================================
1. OVERVIEW & OBJECTIVE:
   Bidirectional validation and serialization mapping for Git Submodule operations:
     - `from_api_add_submodule`: Parses incoming add submodule payload.
     - `from_api_update_submodule`: Parses incoming update submodule payload.
     - `from_api_pull_submodules`: Parses incoming pull submodules payload.
     - `from_api_push_multi_repo`: Parses incoming push multi-repo payload.
     - `from_api_sync_status`: Parses incoming sync status payload.
     - `to_api_result`: Serializes SubmoduleOperationResult to camelCase RFC dict.

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Converts validation exceptions to RFC-standard ValidationError.
================================================================================
"""

from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
    SubmoduleOperationResult,
)


def from_api_add_submodule(payload: Dict[str, Any]) -> Result[AddSubmoduleCommand, ValidationError]:
    try:
        cmd = AddSubmoduleCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for add submodule payload", details=details))


def from_api_update_submodule(payload: Dict[str, Any]) -> Result[UpdateSubmoduleCommand, ValidationError]:
    try:
        cmd = UpdateSubmoduleCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for update submodule payload", details=details))


def from_api_pull_submodules(payload: Dict[str, Any]) -> Result[PullSubmodulesCommand, ValidationError]:
    try:
        cmd = PullSubmodulesCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for pull submodules payload", details=details))


def from_api_push_multi_repo(payload: Dict[str, Any]) -> Result[PushMultiRepoCommand, ValidationError]:
    try:
        cmd = PushMultiRepoCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for push multi-repo payload", details=details))


def from_api_sync_status(payload: Dict[str, Any]) -> Result[SubmoduleSyncStatusCommand, ValidationError]:
    try:
        cmd = SubmoduleSyncStatusCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for submodule sync status payload", details=details))


def to_api_result(result: SubmoduleOperationResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitSubmoduleManagerSchema:
    from_api_add_submodule = staticmethod(from_api_add_submodule)
    from_api_update_submodule = staticmethod(from_api_update_submodule)
    from_api_pull_submodules = staticmethod(from_api_pull_submodules)
    from_api_push_multi_repo = staticmethod(from_api_push_multi_repo)
    from_api_sync_status = staticmethod(from_api_sync_status)
    to_api_result = staticmethod(to_api_result)

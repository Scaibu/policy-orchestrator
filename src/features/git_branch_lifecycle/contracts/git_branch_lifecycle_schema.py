"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE SCHEMA & ANTI-CORRUPTION LAYER (ACL)
================================================================================
1. OVERVIEW & OBJECTIVE:
   Bidirectional validation and serialization mapping for Git branch lifecycle:
     - `from_api_create_branch`: Parses incoming create branch payload.
     - `from_api_delete_branch`: Parses incoming delete branch payload.
     - `from_api_merge_branch`: Parses incoming merge branch payload.
     - `from_api_create_repo`: Parses incoming repository creation payload.
     - `to_api_result`: Serializes BranchLifecycleResult to camelCase RFC dict.

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
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    BranchLifecycleResult,
)


def from_api_create_branch(payload: Dict[str, Any]) -> Result[CreateBranchCommand, ValidationError]:
    try:
        cmd = CreateBranchCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for create branch payload", details=details))


def from_api_delete_branch(payload: Dict[str, Any]) -> Result[DeleteBranchCommand, ValidationError]:
    try:
        cmd = DeleteBranchCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for delete branch payload", details=details))


def from_api_merge_branch(payload: Dict[str, Any]) -> Result[MergeBranchCommand, ValidationError]:
    try:
        cmd = MergeBranchCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for merge branch payload", details=details))


def from_api_create_repo(payload: Dict[str, Any]) -> Result[CreateRepositoryCommand, ValidationError]:
    try:
        cmd = CreateRepositoryCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for create repo payload", details=details))


def to_api_result(result: BranchLifecycleResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitBranchLifecycleSchema:
    from_api_create_branch = staticmethod(from_api_create_branch)
    from_api_delete_branch = staticmethod(from_api_delete_branch)
    from_api_merge_branch = staticmethod(from_api_merge_branch)
    from_api_create_repo = staticmethod(from_api_create_repo)
    to_api_result = staticmethod(to_api_result)

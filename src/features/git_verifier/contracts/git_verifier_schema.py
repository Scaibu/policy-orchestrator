"""
================================================================================
BLUEPRINT: GIT VERIFIER ACL SCHEMA MAPPER
================================================================================
1. OVERVIEW:
   Provides bidirectional Anti-Corruption Layer (ACL) transformations between API
   payloads and domain types for Git state and submodule verification.
================================================================================
"""
from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.git_verifier.contracts.git_verifier_types import (
    VerifyGitCommand,
    GitVerificationResult,
)


def from_api_verify_git(payload: Dict[str, Any]) -> Result[VerifyGitCommand, ValidationError]:
    try:
        cmd = VerifyGitCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for git verify payload", details=details))


def to_api_verify_git(result: GitVerificationResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitVerifierSchema:
    from_api_verify_git = staticmethod(from_api_verify_git)
    to_api_verify_git = staticmethod(to_api_verify_git)

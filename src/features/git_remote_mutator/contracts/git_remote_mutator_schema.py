from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
    RemoteMutationResult,
)


def from_api_mutate_file(payload: Dict[str, Any]) -> Result[MutateFileCommand, ValidationError]:
    try:
        cmd = MutateFileCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for mutate file payload", details=details))


def from_api_mutate_batch(payload: Dict[str, Any]) -> Result[MutateBatchCommand, ValidationError]:
    try:
        cmd = MutateBatchCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for mutate batch payload", details=details))


def from_api_delete_file(payload: Dict[str, Any]) -> Result[DeleteFileCommand, ValidationError]:
    try:
        cmd = DeleteFileCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for delete file payload", details=details))


def to_api_mutation_result(result: RemoteMutationResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitRemoteMutatorSchema:
    from_api_mutate_file = staticmethod(from_api_mutate_file)
    from_api_mutate_batch = staticmethod(from_api_mutate_batch)
    from_api_delete_file = staticmethod(from_api_delete_file)
    to_api_mutation_result = staticmethod(to_api_mutation_result)

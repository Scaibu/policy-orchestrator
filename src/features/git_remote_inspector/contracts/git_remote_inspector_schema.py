from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    RemoteInspectionResult,
    RemoteFileContentResult,
    RemoteDiffResult,
)


def from_api_search_files(payload: Dict[str, Any]) -> Result[SearchFilesCommand, ValidationError]:
    try:
        cmd = SearchFilesCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for search files payload", details=details))


def to_api_search_files(result: RemoteInspectionResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_get_file_content(payload: Dict[str, Any]) -> Result[GetFileContentCommand, ValidationError]:
    try:
        cmd = GetFileContentCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for file content payload", details=details))


def to_api_get_file_content(result: RemoteFileContentResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_get_diff(payload: Dict[str, Any]) -> Result[GetDiffCommand, ValidationError]:
    try:
        cmd = GetDiffCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for get diff payload", details=details))


def to_api_get_diff(result: RemoteDiffResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitRemoteInspectorSchema:
    from_api_search_files = staticmethod(from_api_search_files)
    to_api_search_files = staticmethod(to_api_search_files)
    from_api_get_file_content = staticmethod(from_api_get_file_content)
    to_api_get_file_content = staticmethod(to_api_get_file_content)
    from_api_get_diff = staticmethod(from_api_get_diff)
    to_api_get_diff = staticmethod(to_api_get_diff)

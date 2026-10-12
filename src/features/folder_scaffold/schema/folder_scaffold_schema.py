from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldFeatureCommand,
    ScaffoldPackageCommand,
    CreateFileResult,
    LinkFilesResult,
    ScaffoldExecutionResult,
)


def from_api_create_file(payload: Dict[str, Any]) -> Result[CreateFileCommand, ValidationError]:
    try:
        cmd = CreateFileCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for create file payload", details=details))


def to_api_create_file(result: CreateFileResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_link_files(payload: Dict[str, Any]) -> Result[LinkFilesCommand, ValidationError]:
    try:
        cmd = LinkFilesCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for link files payload", details=details))


def to_api_link_files(result: LinkFilesResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_scaffold_feature(payload: Dict[str, Any]) -> Result[ScaffoldFeatureCommand, ValidationError]:
    try:
        cmd = ScaffoldFeatureCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for scaffold feature payload", details=details))


def from_api_scaffold_package(payload: Dict[str, Any]) -> Result[ScaffoldPackageCommand, ValidationError]:
    try:
        cmd = ScaffoldPackageCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for scaffold package payload", details=details))


def to_api_scaffold_result(result: ScaffoldExecutionResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class FolderScaffoldSchema:
    from_api_create_file = staticmethod(from_api_create_file)
    to_api_create_file = staticmethod(to_api_create_file)
    from_api_link_files = staticmethod(from_api_link_files)
    to_api_link_files = staticmethod(to_api_link_files)
    from_api_scaffold_feature = staticmethod(from_api_scaffold_feature)
    from_api_scaffold_package = staticmethod(from_api_scaffold_package)
    to_api_scaffold_result = staticmethod(to_api_scaffold_result)




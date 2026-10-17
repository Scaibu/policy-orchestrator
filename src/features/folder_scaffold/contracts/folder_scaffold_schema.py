from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    CreateFileResult,
    LinkFilesResult,
    ScaffoldTreeCommand,
    ScaffoldTreeResult,
    BlastRadiusReport,
    FileLineageReport,
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


def from_api_scaffold_tree(payload: Dict[str, Any]) -> Result[ScaffoldTreeCommand, ValidationError]:
    try:
        cmd = ScaffoldTreeCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for scaffold tree payload", details=details))


def to_api_scaffold_tree(result: ScaffoldTreeResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def to_api_blast_radius(report: BlastRadiusReport) -> Dict[str, Any]:
    return dict_to_camel_case(report.model_dump())


def to_api_file_lineage(report: FileLineageReport) -> Dict[str, Any]:
    return dict_to_camel_case(report.model_dump())


class FolderScaffoldSchema:
    from_api_create_file = staticmethod(from_api_create_file)
    to_api_create_file = staticmethod(to_api_create_file)
    from_api_link_files = staticmethod(from_api_link_files)
    to_api_link_files = staticmethod(to_api_link_files)
    from_api_scaffold_tree = staticmethod(from_api_scaffold_tree)
    to_api_scaffold_tree = staticmethod(to_api_scaffold_tree)
    to_api_blast_radius = staticmethod(to_api_blast_radius)
    to_api_file_lineage = staticmethod(to_api_file_lineage)

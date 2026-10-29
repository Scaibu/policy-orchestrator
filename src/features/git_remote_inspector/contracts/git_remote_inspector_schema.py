from typing import Dict, Any
from pydantic import ValidationError as PydanticValidationError
from src.shared.kernel.result import Result
from src.shared.errors.application_error import ValidationError
from src.shared.utils.data_utils import dict_to_camel_case
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    GetTreeHierarchyCommand,
    RemoteFolderHierarchyResult,
    ListBranchesCommand,
    RemoteBranchesResult,
    DiscoverContractsCommand,
    RemoteDiscoveredContractsResult,
    TraceCallChainCommand,
    RemoteCallChainResult,
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


def from_api_get_tree(payload: Dict[str, Any]) -> Result[GetTreeHierarchyCommand, ValidationError]:
    try:
        cmd = GetTreeHierarchyCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for get tree payload", details=details))


def to_api_get_tree(result: RemoteFolderHierarchyResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_list_branches(payload: Dict[str, Any]) -> Result[ListBranchesCommand, ValidationError]:
    try:
        cmd = ListBranchesCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for list branches payload", details=details))


def to_api_list_branches(result: RemoteBranchesResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_discover_contracts(payload: Dict[str, Any]) -> Result[DiscoverContractsCommand, ValidationError]:
    try:
        cmd = DiscoverContractsCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for discover contracts payload", details=details))


def to_api_discover_contracts(result: RemoteDiscoveredContractsResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


def from_api_trace_call_chain(payload: Dict[str, Any]) -> Result[TraceCallChainCommand, ValidationError]:
    try:
        cmd = TraceCallChainCommand.model_validate(payload)
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
        return Result.err(ValidationError(message="Validation failed for trace call chain payload", details=details))


def to_api_trace_call_chain(result: RemoteCallChainResult) -> Dict[str, Any]:
    return dict_to_camel_case(result.model_dump())


class GitRemoteInspectorSchema:
    from_api_search_files = staticmethod(from_api_search_files)
    to_api_search_files = staticmethod(to_api_search_files)
    from_api_get_file_content = staticmethod(from_api_get_file_content)
    to_api_get_file_content = staticmethod(to_api_get_file_content)
    from_api_get_diff = staticmethod(from_api_get_diff)
    to_api_get_diff = staticmethod(to_api_get_diff)
    from_api_get_tree = staticmethod(from_api_get_tree)
    to_api_get_tree = staticmethod(to_api_get_tree)
    from_api_list_branches = staticmethod(from_api_list_branches)
    to_api_list_branches = staticmethod(to_api_list_branches)
    from_api_discover_contracts = staticmethod(from_api_discover_contracts)
    to_api_discover_contracts = staticmethod(to_api_discover_contracts)
    from_api_trace_call_chain = staticmethod(from_api_trace_call_chain)
    to_api_trace_call_chain = staticmethod(to_api_trace_call_chain)


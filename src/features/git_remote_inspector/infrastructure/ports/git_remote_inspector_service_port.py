from abc import ABC, abstractmethod
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    RemoteInspectionResult,
    RemoteFileContentResult,
    RemoteDiffResult,
)


class GitRemoteInspectorServicePort(ABC):
    @abstractmethod
    def search_files(
        self, ctx: TraceContext, cmd: SearchFilesCommand
    ) -> Result[RemoteInspectionResult, Exception]:
        pass

    @abstractmethod
    def get_file_content(
        self, ctx: TraceContext, cmd: GetFileContentCommand
    ) -> Result[RemoteFileContentResult, Exception]:
        pass

    @abstractmethod
    def get_diff(
        self, ctx: TraceContext, cmd: GetDiffCommand
    ) -> Result[RemoteDiffResult, Exception]:
        pass

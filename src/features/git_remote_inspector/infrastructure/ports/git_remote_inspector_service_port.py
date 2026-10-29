from abc import ABC, abstractmethod
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    GetTreeHierarchyCommand,
    ListBranchesCommand,
    DiscoverContractsCommand,
    TraceCallChainCommand,
    RemoteInspectionResult,
    RemoteFileContentResult,
    RemoteDiffResult,
    RemoteFolderHierarchyResult,
    RemoteBranchesResult,
    RemoteDiscoveredContractsResult,
    RemoteCallChainResult,
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

    @abstractmethod
    def get_tree_hierarchy(
        self, ctx: TraceContext, cmd: GetTreeHierarchyCommand
    ) -> Result[RemoteFolderHierarchyResult, Exception]:
        pass

    @abstractmethod
    def list_branches(
        self, ctx: TraceContext, cmd: ListBranchesCommand
    ) -> Result[RemoteBranchesResult, Exception]:
        pass

    @abstractmethod
    def discover_contracts(
        self, ctx: TraceContext, cmd: DiscoverContractsCommand
    ) -> Result[RemoteDiscoveredContractsResult, Exception]:
        pass

    @abstractmethod
    def trace_call_chain(
        self, ctx: TraceContext, cmd: TraceCallChainCommand
    ) -> Result[RemoteCallChainResult, Exception]:
        pass


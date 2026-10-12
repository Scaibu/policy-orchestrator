from abc import ABC, abstractmethod
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldFeatureCommand,
    ScaffoldPackageCommand,
    CreateFileResult,
    LinkFilesResult,
    ScaffoldExecutionResult,
)


class FolderScaffoldServicePort(ABC):
    @abstractmethod
    def createFile(
        self, ctx: TraceContext, cmd: CreateFileCommand
    ) -> Result[CreateFileResult, Exception]:
        pass

    @abstractmethod
    def linkFiles(
        self, ctx: TraceContext, cmd: LinkFilesCommand
    ) -> Result[LinkFilesResult, Exception]:
        pass

    @abstractmethod
    def scaffoldFeature(
        self, ctx: TraceContext, cmd: ScaffoldFeatureCommand
    ) -> Result[ScaffoldExecutionResult, Exception]:
        pass

    @abstractmethod
    def scaffoldPackage(
        self, ctx: TraceContext, cmd: ScaffoldPackageCommand
    ) -> Result[ScaffoldExecutionResult, Exception]:
        pass

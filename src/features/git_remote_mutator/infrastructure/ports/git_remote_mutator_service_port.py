from abc import ABC, abstractmethod
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
    RemoteMutationResult,
)


class GitRemoteMutatorServicePort(ABC):
    @abstractmethod
    def mutate_file(
        self, ctx: TraceContext, cmd: MutateFileCommand
    ) -> Result[RemoteMutationResult, Exception]:
        pass

    @abstractmethod
    def mutate_batch(
        self, ctx: TraceContext, cmd: MutateBatchCommand
    ) -> Result[RemoteMutationResult, Exception]:
        pass

    @abstractmethod
    def delete_file(
        self, ctx: TraceContext, cmd: DeleteFileCommand
    ) -> Result[RemoteMutationResult, Exception]:
        pass

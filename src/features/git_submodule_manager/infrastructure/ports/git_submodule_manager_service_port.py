"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER SERVICE PORT
================================================================================
1. OVERVIEW & OBJECTIVE:
   Abstract driving port contract for multi-repo submodule management.
   Exposed directly to delivery handlers and CLI.

2. INVARIANTS:
   - ZERO comments inside method bodies.
   - Return Result[T, Exception] wrappers.
================================================================================
"""

from abc import ABC, abstractmethod
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
    SubmoduleOperationResult,
)


class GitSubmoduleManagerServicePort(ABC):
    @abstractmethod
    def add_submodule(
        self, ctx: TraceContext, cmd: AddSubmoduleCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def update_submodule(
        self, ctx: TraceContext, cmd: UpdateSubmoduleCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def pull_submodules(
        self, ctx: TraceContext, cmd: PullSubmodulesCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def push_multi_repo(
        self, ctx: TraceContext, cmd: PushMultiRepoCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def get_sync_status(
        self, ctx: TraceContext, cmd: SubmoduleSyncStatusCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        raise NotImplementedError

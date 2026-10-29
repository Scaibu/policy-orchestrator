"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE DRIVING SERVICE PORT
================================================================================
1. OVERVIEW & OBJECTIVE:
   Abstract driving port contract for branch lifecycle features.
   Exposed directly to delivery handlers and CLI.

2. INVARIANTS:
   - ZERO comments inside method bodies.
   - Return Result[T, Exception] wrappers.
================================================================================
"""

from abc import ABC, abstractmethod
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    BranchLifecycleResult,
)


class GitBranchLifecycleServicePort(ABC):
    @abstractmethod
    def create_branch(
        self, ctx: TraceContext, cmd: CreateBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def delete_branch(
        self, ctx: TraceContext, cmd: DeleteBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def merge_branch(
        self, ctx: TraceContext, cmd: MergeBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        raise NotImplementedError

    @abstractmethod
    def create_repository(
        self, ctx: TraceContext, cmd: CreateRepositoryCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        raise NotImplementedError

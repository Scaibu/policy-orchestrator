"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE PUBLIC FACADE
================================================================================
1. OVERVIEW & OBJECTIVE:
   Role (13) public facade exporting domain service and public DTOs only.
   Shields internal adapters and queries from outside consumers.

2. INVARIANTS:
   - ZERO comments inside code bodies.
================================================================================
"""

from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    BranchLifecycleResult,
    GitBranchLifecycleBaseError,
    BranchAlreadyExistsError,
    BranchNotFoundError,
    MergeConflictError,
    InvalidBranchNameError,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_deps import (
    GitBranchLifecycleServiceDeps,
)
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_service_port import (
    GitBranchLifecycleServicePort,
)
from src.features.git_branch_lifecycle.service.git_branch_lifecycle_service import (
    GitBranchLifecycleDomainService,
)

__all__ = [
    "CreateBranchCommand",
    "DeleteBranchCommand",
    "MergeBranchCommand",
    "CreateRepositoryCommand",
    "BranchLifecycleResult",
    "GitBranchLifecycleBaseError",
    "BranchAlreadyExistsError",
    "BranchNotFoundError",
    "MergeConflictError",
    "InvalidBranchNameError",
    "GitBranchLifecycleServiceDeps",
    "GitBranchLifecycleServicePort",
    "GitBranchLifecycleDomainService",
]

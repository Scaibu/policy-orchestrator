"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE SERVICE DEPENDENCIES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Dependency injection container for GitBranchLifecycleDomainService.

2. INVARIANTS:
   - ZERO comments inside class bodies.
================================================================================
"""

from typing import Dict, Any
from dataclasses import dataclass
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_repository_port import (
    GitBranchLifecycleRepositoryPort,
)
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_client_port import (
    GitBranchLifecycleClientPort,
)


@dataclass(frozen=True)
class GitBranchLifecycleServiceDeps:
    repository: GitBranchLifecycleRepositoryPort
    client: GitBranchLifecycleClientPort
    config: Dict[str, Any]

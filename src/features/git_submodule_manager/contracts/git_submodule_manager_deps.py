"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER SERVICE DEPENDENCIES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Dependency injection container for GitSubmoduleManagerDomainService.

2. INVARIANTS:
   - ZERO comments inside class bodies.
================================================================================
"""

from typing import Dict, Any
from dataclasses import dataclass, field
from src.features.git_submodule_manager.infrastructure.ports.git_submodule_manager_repository_port import (
    GitSubmoduleManagerRepositoryPort,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)


@dataclass(frozen=True)
class GitSubmoduleManagerServiceDeps:
    repository: GitSubmoduleManagerRepositoryPort
    mutator_client: GitRemoteMutationClientPort
    inspector_client: GitRemoteClientPort
    config: Dict[str, Any] = field(default_factory=dict)

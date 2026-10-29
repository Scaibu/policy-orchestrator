from dataclasses import dataclass, field
from typing import Dict, Any
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutator_repository_port import (
    GitRemoteMutatorRepositoryPort,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)


@dataclass(frozen=True)
class GitRemoteMutatorServiceDeps:
    repository: GitRemoteMutatorRepositoryPort
    mutation_client: GitRemoteMutationClientPort
    config: Dict[str, Any] = field(default_factory=dict)

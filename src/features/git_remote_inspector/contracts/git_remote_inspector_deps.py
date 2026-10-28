from dataclasses import dataclass, field
from typing import Dict, Any
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_repository_port import (
    GitRemoteInspectorRepositoryPort,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)


@dataclass(frozen=True)
class GitRemoteInspectorServiceDeps:
    repository: GitRemoteInspectorRepositoryPort
    remote_client: GitRemoteClientPort
    config: Dict[str, Any] = field(default_factory=dict)

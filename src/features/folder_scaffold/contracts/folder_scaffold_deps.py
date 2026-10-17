from dataclasses import dataclass, field
from typing import Dict, Any
from src.features.folder_scaffold.infrastructure.ports.folder_scaffold_repository_port import FolderScaffoldRepositoryPort
from src.domain.ports.graph_port import GraphStorePort


@dataclass(frozen=True)
class FolderScaffoldServiceDeps:
    repository: FolderScaffoldRepositoryPort
    graph_store: GraphStorePort
    scaffold_config: Dict[str, Any] = field(default_factory=dict)

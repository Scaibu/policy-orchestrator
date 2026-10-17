from abc import ABC, abstractmethod
from typing import Optional
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    ScaffoldedFileNode,
    ScaffoldedRelationshipRecord,
)


class FolderScaffoldRepositoryPort(ABC):
    @abstractmethod
    def save_node(self, node: ScaffoldedFileNode) -> None:
        pass

    @abstractmethod
    def save_relationship(self, relationship: ScaffoldedRelationshipRecord) -> None:
        pass

    @abstractmethod
    def find_node_by_id(self, node_id: str) -> Optional[ScaffoldedFileNode]:
        pass

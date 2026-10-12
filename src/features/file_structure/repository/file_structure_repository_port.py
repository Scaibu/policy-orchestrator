from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class FileStructureRepositoryPort(ABC):
    @abstractmethod
    def save_node(
        self,
        node_id: str,
        label: str,
        file_path: str,
        feature_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def list_upstream_dependencies(self, target_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def list_downstream_dependents(self, source_id: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def clear_all(self) -> None:
        pass

    @abstractmethod
    def get_scaffold_template(
        self,
        template_type: str,
        version: str = "v1",
    ) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_scaffold_template(
        self,
        template_id: str,
        template_type: str,
        version: str,
        directories: List[str],
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def save_scaffolded_feature(
        self,
        feature_name: str,
        target_dir: str,
        port_type: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def save_scaffolded_package(
        self,
        package_name: str,
        target_root: str,
        directories: List[str],
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def seed_default_templates(self) -> None:
        pass

KnowledgeGraphRepositoryPort = FileStructureRepositoryPort

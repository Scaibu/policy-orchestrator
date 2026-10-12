from typing import Dict, Any, Optional
from src.domain.ports.graph_port import GraphStorePort
from src.domain.ports.impact_analysis_port import ImpactAnalysisReportModel
from src.features.file_structure.machines.file_structure_machine import FileStructureLifecycleMachine
from src.features.file_structure.service.file_structure_folder_scaffolder_service import FileStructureFolderScaffolderService
from src.features.file_structure.service.file_structure_package_scaffolder_service import FileStructurePackageScaffolderService
from src.features.file_structure.service.file_structure_scaffolder_service import FileStructureScaffolderService
from src.features.file_structure.service.file_structure_graph_sync_service import FileStructureGraphSyncService
from src.features.file_structure.service.file_structure_impact_analyzer_service import FileStructureImpactAnalyzerService
from src.features.file_structure.service.file_structure_map_resolver_service import FileStructureMapResolverService

class FileStructureDomainService:
    def __init__(
        self,
        graph_store: GraphStorePort,
    ) -> None:
        self.graph_store = graph_store
        self.folder_scaffolder = FileStructureFolderScaffolderService()
        self.package_scaffolder = FileStructurePackageScaffolderService()
        self.scaffolder = FileStructureScaffolderService(
            folder_scaffolder=self.folder_scaffolder,
            package_scaffolder=self.package_scaffolder,
        )
        self.graph_sync = FileStructureGraphSyncService(graph_store=graph_store)
        self.impact_analyzer = FileStructureImpactAnalyzerService(graph_store=graph_store)
        self.map_resolver = FileStructureMapResolverService(graph_store=graph_store)
        self.state_machine = FileStructureLifecycleMachine()

    def scaffold_feature(
        self,
        feature_name: str,
        base_dir: str = "src/features",
        package_root: Optional[str] = None,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
        port_type: str = "local",
        owner: Optional[str] = None,
        status: str = "active",
        flags: Optional[str] = None,
        migrations: Optional[str] = None,
        contract_version: str = "v1",
    ) -> Dict[str, Any]:
        self.state_machine.transition_to("SCAFFOLDING")
        try:
            tree_result = self.folder_scaffolder.scaffold_feature_tree(
                feature_name=feature_name,
                base_dir=base_dir,
                package_root=package_root,
                with_router=with_router,
                with_handler=with_handler,
                with_route_rules=with_route_rules,
                port_type=port_type,
            )
            self.graph_sync.sync_feature_dag(
                feature_name=feature_name,
                with_router=with_router,
                with_handler=with_handler,
                with_route_rules=with_route_rules,
                port_type=port_type,
                package_root=package_root,
                owner=owner,
                status=status,
                flags=flags,
                migrations=migrations,
                contract_version=contract_version,
            )
            self.state_machine.transition_to("INDEXED")
            tree_result["graph_synced"] = True
            return tree_result
        except Exception as exc:
            self.state_machine.transition_to("ERROR")
            raise exc

    def scaffold_package(self, package_name: str, base_dir: str = ".") -> Dict[str, Any]:
        self.state_machine.transition_to("SCAFFOLDING")
        try:
            pkg_result = self.package_scaffolder.scaffold_package_tree(
                package_name=package_name,
                base_dir=base_dir,
            )
            self.graph_sync.sync_package_dag(
                package_name=package_name,
                target_root=pkg_result["target_root"],
            )
            self.state_machine.transition_to("INDEXED")
            pkg_result["graph_synced"] = True
            return pkg_result
        except Exception as exc:
            self.state_machine.transition_to("ERROR")
            raise exc

    def get_feature_map(self, feature_name: str) -> Dict[str, Any]:
        return self.map_resolver.resolve_feature_map(feature_name=feature_name)

    def get_package_summary(self, package_name: str) -> Dict[str, Any]:
        return self.map_resolver.resolve_package_summary(package_name=package_name)

    def get_feature_files(self, feature_name: str, package_name: Optional[str] = None) -> Dict[str, Any]:
        return self.map_resolver.resolve_feature_files(feature_name=feature_name, package_name=package_name)

    def analyze_node_impact(
        self,
        target_id: str,
        direction: str = "UPSTREAM",
        max_depth: int = 6,
    ) -> ImpactAnalysisReportModel:
        if not target_id:
            raise ValueError("target_id must not be empty")

        if direction.upper() == "UPSTREAM":
            return self.impact_analyzer.analyze_upstream_impact(target_id=target_id, max_depth=max_depth)
        elif direction.upper() == "DOWNSTREAM":
            return self.impact_analyzer.analyze_downstream_impact(target_id=target_id, max_depth=max_depth)
        else:
            raise ValueError(f"Unsupported direction: {direction}")

    def scan_and_sync_workspace(self, root_dir: str) -> Dict[str, Any]:
        self.state_machine.transition_to("SCANNING")
        try:
            stats = self.graph_sync.scan_and_index_repository(root_dir=root_dir)
            self.state_machine.transition_to("INDEXED")
            return {
                "status": "success",
                "state": self.state_machine.current_state,
                "stats": stats,
            }
        except Exception as exc:
            self.state_machine.transition_to("ERROR")
            raise exc

    def get_file_lineage(self, file_path_or_node_id: str, max_depth: int = 5) -> Dict[str, Any]:
        return self.map_resolver.resolve_file_lineage(file_path_or_node_id=file_path_or_node_id, max_depth=max_depth)

    def create_file_with_relationship(
        self,
        file_path: str,
        role: str,
        rel_type: str,
        target_file_or_node: str,
        direction: str = "outgoing",
        content: Optional[str] = None,
        package_root: Optional[str] = None,
        feature_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not rel_type:
            raise ValueError("rel_type is a required parameter: every newly created file must have an architectural relationship")
        if not target_file_or_node:
            raise ValueError("target_file_or_node is a required parameter: specify the upstream/downstream file or node to connect")

        created_path = self.folder_scaffolder.create_file(file_path=file_path, role=role, content=content)
        sync_result = self.graph_sync.register_and_link_file(
            file_path=created_path,
            role=role,
            rel_type=rel_type,
            target_file_or_node=target_file_or_node,
            direction=direction,
            package_root=package_root,
            feature_name=feature_name,
            properties=properties,
        )

        return {
            "status": "success",
            "file_path": created_path,
            "role": role,
            "graph_sync": sync_result,
        }

    def link_files(
        self,
        source: str,
        rel_type: str,
        target: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.graph_sync.link_files(
            source_file_or_node=source,
            rel_type=rel_type,
            target_file_or_node=target,
            properties=properties,
        )

    def link_migration(self, feature_name: str, migration_file: str) -> None:
        self.graph_sync.link_database_migration(feature_name=feature_name, migration_file=migration_file)

    def link_kafka_event(
        self,
        feature_name: str,
        event_name: str,
        topic_name: str,
        topic_spec_path: Optional[str] = None,
    ) -> None:
        self.graph_sync.link_kafka_event_topic(
            feature_name=feature_name,
            event_name=event_name,
            topic_name=topic_name,
            topic_spec_path=topic_spec_path,
        )

    def link_cross_feature_event(
        self,
        publisher_feature: str,
        consumer_feature: str,
        event_name: str,
        topic_name: str,
    ) -> None:
        self.graph_sync.link_cross_feature_event(
            publisher_feature=publisher_feature,
            consumer_feature=consumer_feature,
            event_name=event_name,
            topic_name=topic_name,
        )

    def link_shared(self, feature_name: str, shared_file: str) -> None:
        self.graph_sync.link_shared_dependency(feature_name=feature_name, shared_file=shared_file)

KnowledgeGraphDomainService = FileStructureDomainService
FileStructureService = FileStructureDomainService

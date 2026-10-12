
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
from src.domain.ports.graph_port import (
    GraphStorePort,
    GraphNode,
    GraphRelationship,
)
from src.infra.filesystem.file_walker import stream_repository_files
from src.features.file_structure.rules.file_structure_role_rules import RoleLayerRegistry
from src.features.file_structure.rules.file_structure_direction_rules import EdgeDirectionRegistry
from src.features.file_structure.rules.file_structure_path_rules import PathLabelRuleEngine
from src.features.file_structure.rules.file_structure_validation_rules import FileStructureValidationRuleEngine
from src.features.file_structure.rules.file_structure_slug_pipeline import FeatureSlugPipeline
from src.features.file_structure.schema.file_structure_graph_mapper import FeatureGraphMapper

class FileStructureGraphSyncService:
    def __init__(self, graph_store: GraphStorePort) -> None:
        self.graph_store = graph_store

    def register_node(
        self,
        node_id: str,
        label: str,
        file_path: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        props = properties or {}
        props["file_path"] = file_path
        props["node_id"] = node_id
        props["label"] = label
        node = GraphNode(id=node_id, label=label, properties=props)
        self.graph_store.upsert_node(node)

    def register_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        rel = GraphRelationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=relationship_type,
            properties=properties or {},
        )
        self.graph_store.upsert_relationship(rel)

    def sync_feature_dag(
        self,
        feature_name: str,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
        port_type: str = "local",
        package_root: Optional[str] = None,
        owner: Optional[str] = None,
        status: str = "active",
        flags: Optional[str] = None,
        migrations: Optional[str] = None,
        contract_version: str = "v1",
    ) -> None:
        ctx = FeatureGraphMapper.map_to_graph_context(
            feature_name=feature_name,
            port_type=port_type,
            package_root=package_root,
            owner=owner,
            status=status,
            flags=flags,
            migrations=migrations,
            contract_version=contract_version,
        )

        node_specs = FeatureGraphMapper.map_to_node_specs(
            ctx=ctx,
            port_type=port_type,
            with_router=with_router,
            with_handler=with_handler,
            with_route_rules=with_route_rules,
        )
        props = ctx.to_properties_dict()
        for node in node_specs:
            self.register_node(node.id, node.label, node.file_path, {**props, "layer": node.layer})

        edges = FeatureGraphMapper.map_to_edge_specs(
            ctx=ctx,
            with_router=with_router,
            with_handler=with_handler,
            with_route_rules=with_route_rules,
        )
        for src, tgt, rel in edges:
            self.register_edge(src, tgt, rel)

    def sync_package_dag(self, package_name: str, target_root: str) -> None:
        pkg_slug = package_name.lower().replace("-", "_")
        pkg_node_id = f"pkg_{pkg_slug}"
        self.register_node(
            node_id=pkg_node_id,
            label="PackageRoot",
            file_path=target_root,
            properties={"name": pkg_slug},
        )

    def link_database_migration(self, feature_name: str, migration_file: str) -> None:
        f = feature_name.lower().replace("-", "_")
        migration_id = f"migration_{Path(migration_file).stem}"
        query_id = f"query_{f}_sql"
        self.register_node(migration_id, "DatabaseMigration", migration_file)
        self.register_edge(query_id, migration_id, "DEPENDS_ON_DDL")

    def link_kafka_event_topic(
        self,
        feature_name: str,
        event_name: str,
        topic_name: str,
        topic_spec_path: Optional[str] = None,
    ) -> None:
        f = feature_name.lower().replace("-", "_")
        service_id = f"service_{f}"
        producer_id = f"producer_{f}_{event_name.lower()}"
        topic_id = f"topic_{topic_name.replace('.', '_').replace('-', '_')}"

        self.register_node(producer_id, "EventProducer", f"src/features/{f}/repository/{f}.event.producer.py")
        self.register_node(topic_id, "TopicSpec", topic_spec_path or f"messaging/topics/{topic_name}.yaml")

        self.register_edge(service_id, producer_id, "EMITS_EVENT")
        self.register_edge(producer_id, topic_id, "PUBLISHES_TO")

    def link_cross_feature_event(
        self,
        publisher_feature: str,
        consumer_feature: str,
        event_name: str,
        topic_name: str,
    ) -> None:
        pub_f = publisher_feature.lower().replace("-", "_")
        sub_f = consumer_feature.lower().replace("-", "_")
        topic_id = f"topic_{topic_name.replace('.', '_').replace('-', '_')}"
        consumer_id = f"consumer_{sub_f}_{event_name.lower()}"
        sub_service_id = f"service_{sub_f}"

        self.link_kafka_event_topic(publisher_feature, event_name, topic_name)
        self.register_node(consumer_id, "EventConsumer", f"src/api/events/consumers/{sub_f}_{event_name.lower()}_consumer.py")

        self.register_edge(topic_id, consumer_id, "DELIVERS_TO")
        self.register_edge(consumer_id, sub_service_id, "TRIGGERS_DOMAIN")

    def link_shared_dependency(self, feature_name: str, shared_file: str) -> None:
        f = feature_name.lower().replace("-", "_")
        service_id = f"service_{f}"
        shared_id = f"shared_{Path(shared_file).stem}"
        self.register_node(shared_id, "SharedLibrary", shared_file)
        self.register_edge(service_id, shared_id, "IMPORTS_SHARED")

    def scan_and_index_repository(
        self,
        root_dir: str,
        allowed_extensions: Optional[Set[str]] = None,
        auto_sync_features: bool = True,
    ) -> Dict[str, Any]:
        exts = allowed_extensions or {".py", ".sql", ".yaml", ".yml", ".json", ".graphql", ".proto", ".sh", ".md"}
        indexed_count = 0
        root_path = Path(root_dir).resolve()
        discovered_features: Set[str] = set()
        feature_packages: Dict[str, Optional[str]] = {}

        for fpath in stream_repository_files(root_dir=root_dir, allowed_extensions=exts):
            try:
                rel_path = str(fpath.relative_to(root_path))
            except ValueError:
                rel_path = str(fpath)
            node_id = f"file_{rel_path.replace('/', '_').replace('.', '_')}"
            label = self._infer_label_from_path(rel_path)

            self.register_node(
                node_id=node_id,
                label=label,
                file_path=rel_path,
                properties={
                    "absolute_path": str(fpath),
                    "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
                    "extension": fpath.suffix,
                }
            )
            indexed_count += 1

            parts = Path(rel_path).parts
            if "features" in parts:
                idx = parts.index("features")
                if idx + 1 < len(parts):
                    feat_slug = parts[idx + 1].lower().replace("-", "_")
                    discovered_features.add(feat_slug)
                    if "packages" in parts:
                        pkg_idx = parts.index("packages")
                        if pkg_idx + 1 < idx:
                            feature_packages[feat_slug] = "/".join(parts[:pkg_idx + 2])
                    else:
                        feature_packages.setdefault(feat_slug, None)

        synced_feature_count = 0
        if auto_sync_features:
            for feat in discovered_features:
                pkg_root = feature_packages.get(feat)
                if pkg_root:
                    self.sync_package_dag(package_name=Path(pkg_root).name, target_root=pkg_root)
                self.sync_feature_dag(
                    feature_name=feat,
                    package_root=pkg_root,
                )
                synced_feature_count += 1

        return {
            "indexed_files": indexed_count,
            "discovered_features": len(discovered_features),
            "features": sorted(list(discovered_features)),
            "synced_features_count": synced_feature_count,
            "status": "COMPLETED",
        }

    def register_and_link_file(
        self,
        file_path: str,
        role: str,
        rel_type: str,
        target_file_or_node: str,
        direction: str = "outgoing",
        package_root: Optional[str] = None,
        feature_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        FileStructureValidationRuleEngine.validate_create_file({
            "file_path": file_path,
            "role": role,
            "rel_type": rel_type,
            "target_file_or_node": target_file_or_node,
        })

        p_root = f"{package_root.rstrip('/')}/" if package_root else ""
        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""

        f_slug = FeatureSlugPipeline.resolve_slug(feature_name, file_path)

        stem = Path(file_path).stem.lower().replace("-", "_")
        node_id = f"{p_prefix}{role.lower()}_{stem}"

        props = properties or {}
        if f_slug:
            props["feature_name"] = f_slug
        if pkg_slug:
            props["package_name"] = pkg_slug
        if "layer" not in props:
            props["layer"] = RoleLayerRegistry.resolve(role)

        self.register_node(node_id=node_id, label=role, file_path=file_path, properties=props)

        resolved_target_id = self._resolve_target_node_id(target_file_or_node, package_root=package_root, feature_name=f_slug)

        src_id, tgt_id = EdgeDirectionRegistry.resolve(direction, node_id, resolved_target_id)

        self.register_edge(source_id=src_id, target_id=tgt_id, relationship_type=rel_type)

        return {
            "node_id": node_id,
            "label": role,
            "file_path": file_path,
            "relationship": {
                "source_id": src_id,
                "target_id": tgt_id,
                "rel_type": rel_type,
                "direction": direction,
            }
        }

    def link_files(
        self,
        source_file_or_node: str,
        rel_type: str,
        target_file_or_node: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        FileStructureValidationRuleEngine.validate_link_files({
            "source_file_or_node": source_file_or_node,
            "target_file_or_node": target_file_or_node,
            "rel_type": rel_type,
        })

        src_id = self._resolve_target_node_id(source_file_or_node)
        tgt_id = self._resolve_target_node_id(target_file_or_node)

        self.register_edge(source_id=src_id, target_id=tgt_id, relationship_type=rel_type, properties=properties)

        return {
            "source_id": src_id,
            "target_id": tgt_id,
            "relationship_type": rel_type,
            "linked": True,
        }

    def _resolve_target_node_id(self, target_query: str, package_root: Optional[str] = None, feature_name: Optional[str] = None) -> str:
        q = target_query.strip()
        node = self.graph_store.find_node(q)
        if node:
            return node.id

        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""
        return f"{p_prefix}{q.replace('/', '_').replace('.', '_')}"

    def _infer_layer_from_role(self, role: str) -> str:
        return RoleLayerRegistry.resolve(role)

    def _infer_label_from_path(self, path: str) -> str:
        return PathLabelRuleEngine.resolve(path)


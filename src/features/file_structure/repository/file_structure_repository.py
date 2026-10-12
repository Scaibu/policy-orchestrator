import json
from typing import List, Dict, Any, Optional
from src.features.file_structure.repository.file_structure_repository_port import FileStructureRepositoryPort
from src.domain.ports.graph_port import GraphStorePort, GraphNode, GraphRelationship
from src.features.file_structure.repository.file_structure_query_loader import FileStructureQueryLoader
from src.infra.config.config_loader import AppConfigLoader
from src.shared.constants.scaffold_constants import (
    TEMPLATE_TYPE_FEATURE,
    TEMPLATE_TYPE_PACKAGE,
    DEFAULT_TEMPLATE_VERSION,
)

class FileStructureRepositoryAdapter(FileStructureRepositoryPort):
    def __init__(
        self,
        graph_store: Optional[GraphStorePort] = None,
        db_url: Optional[str] = None,
    ) -> None:
        self.graph_store = graph_store
        self.db_url = db_url or AppConfigLoader.get_database_url()

    def _get_pg_connection(self):
        import psycopg2
        return psycopg2.connect(self.db_url, connect_timeout=3)

    def save_node(
        self,
        node_id: str,
        label: str,
        file_path: str,
        feature_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.graph_store:
            props = properties or {}
            props["feature_name"] = feature_name
            props["file_path"] = file_path
            node = GraphNode(id=node_id, label=label, properties=props)
            self.graph_store.upsert_node(node)

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not self.graph_store:
            return None
        node = self.graph_store.get_node(node_id)
        if not node:
            return None
        return {
            "id": node.id,
            "label": node.label,
            "properties": node.properties,
        }

    def save_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.graph_store:
            rel = GraphRelationship(
                source_id=source_id,
                target_id=target_id,
                rel_type=relationship_type,
                properties=properties or {},
            )
            self.graph_store.upsert_relationship(rel)

    def list_upstream_dependencies(self, target_id: str) -> List[Dict[str, Any]]:
        if not self.graph_store:
            return []
        query = FileStructureQueryLoader.get_cypher("FLOW_LIST_UPSTREAM_DEPENDENCIES_BY_TARGET_ID_CYPHER")
        res = self.graph_store.query_cypher(query, {"target_id": target_id, "id": target_id})
        return res.records

    def list_downstream_dependents(self, source_id: str) -> List[Dict[str, Any]]:
        if not self.graph_store:
            return []
        query = FileStructureQueryLoader.get_cypher("FLOW_LIST_DOWNSTREAM_DEPENDENTS_BY_SOURCE_ID_CYPHER")
        res = self.graph_store.query_cypher(query, {"source_id": source_id, "id": source_id})
        return res.records

    def clear_all(self) -> None:
        if self.graph_store:
            self.graph_store.clear()

    def get_scaffold_template(
        self,
        template_type: str,
        version: str = DEFAULT_TEMPLATE_VERSION,
    ) -> Optional[Dict[str, Any]]:
        raw_query = FileStructureQueryLoader.get_sql("FLOW_GET_SCAFFOLD_TEMPLATE_BY_TYPE_AND_VERSION")
        pg_query = raw_query.replace(":template_type", "%s").replace(":version", "%s")
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                cur.execute(pg_query, (template_type.upper(), version))
                row = cur.fetchone()
                if not row:
                    self.seed_default_templates()
                    cur.execute(pg_query, (template_type.upper(), version))
                    row = cur.fetchone()
                if row:
                    dirs = row[3] if isinstance(row[3], list) else json.loads(row[3])
                    fls = row[4] if isinstance(row[4], list) else json.loads(row[4])
                    meta = row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}")
                    conn.close()
                    return {
                        "id": row[0],
                        "template_type": row[1],
                        "version": row[2],
                        "directories": dirs,
                        "files": fls,
                        "metadata": meta,
                    }
            conn.close()
        except Exception:
            pass
        return None

    def save_scaffold_template(
        self,
        template_id: str,
        template_type: str,
        version: str,
        directories: List[str],
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_query = FileStructureQueryLoader.get_sql("FLOW_INSERT_SCAFFOLD_TEMPLATE")
        pg_query = (
            raw_query.replace(":id", "%s")
            .replace(":template_type", "%s")
            .replace(":version", "%s")
            .replace(":directories", "%s")
            .replace(":files", "%s")
            .replace(":metadata", "%s")
            .replace(":is_active", "%s")
        )
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                cur.execute(
                    pg_query,
                    (
                        template_id,
                        template_type.upper(),
                        version,
                        json.dumps(directories),
                        json.dumps(files),
                        json.dumps(metadata or {}),
                        True,
                    ),
                )
                conn.commit()
            conn.close()
        except Exception:
            pass

    def save_scaffolded_feature(
        self,
        feature_name: str,
        target_dir: str,
        port_type: str,
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_query = FileStructureQueryLoader.get_sql("FLOW_INSERT_SCAFFOLDED_FEATURE")
        pg_query = (
            raw_query.replace(":feature_name", "%s")
            .replace(":target_dir", "%s")
            .replace(":port_type", "%s")
            .replace(":files", "%s")
            .replace(":metadata", "%s")
        )
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                cur.execute(
                    pg_query,
                    (
                        feature_name,
                        target_dir,
                        port_type,
                        json.dumps(files),
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
            conn.close()
        except Exception:
            pass

    def save_scaffolded_package(
        self,
        package_name: str,
        target_root: str,
        directories: List[str],
        files: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_query = FileStructureQueryLoader.get_sql("FLOW_INSERT_SCAFFOLDED_PACKAGE")
        pg_query = (
            raw_query.replace(":package_name", "%s")
            .replace(":target_root", "%s")
            .replace(":directories", "%s")
            .replace(":files", "%s")
            .replace(":metadata", "%s")
        )
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                cur.execute(
                    pg_query,
                    (
                        package_name,
                        target_root,
                        json.dumps(directories),
                        json.dumps(files),
                        json.dumps(metadata or {}),
                    ),
                )
                conn.commit()
            conn.close()
        except Exception:
            pass

    def seed_default_templates(self) -> None:
        seed_query = FileStructureQueryLoader.get_sql("FLOW_SEED_SCAFFOLD_TEMPLATES")
        try:
            conn = self._get_pg_connection()
            with conn.cursor() as cur:
                cur.execute(seed_query)
                conn.commit()
            conn.close()
        except Exception:
            pass

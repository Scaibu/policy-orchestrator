import json
from typing import Optional
from src.features.folder_scaffold.infrastructure.ports.folder_scaffold_repository_port import FolderScaffoldRepositoryPort
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    ScaffoldedFileNode,
    ScaffoldedRelationshipRecord,
)
from src.features.folder_scaffold.infrastructure.queries.folder_scaffold_queries import (
    FLOW_INSERT_SCAFFOLDED_NODE,
    FLOW_INSERT_SCAFFOLDED_RELATIONSHIP,
    FLOW_GET_SCAFFOLDED_NODE_BY_ID,
)


class AlloyDBFolderScaffoldRepository(FolderScaffoldRepositoryPort):
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url.replace("@localhost:", "@127.0.0.1:")
        self._conn = None

    def _get_connection(self):
        import psycopg2
        if self._conn is not None:
            try:
                if self._conn.closed == 0:
                    with self._conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    return self._conn
            except Exception:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def save_node(self, node: ScaffoldedFileNode) -> None:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_SCAFFOLDED_NODE,
                    {
                        "id": node.id,
                        "file_path": node.file_path,
                        "role": node.role,
                        "layer": node.layer,
                        "feature_name": node.feature_name,
                        "package_name": node.package_name,
                        "file_size_bytes": node.size_bytes,
                    },
                )
            conn.commit()
        except Exception:
            pass

    def save_relationship(self, relationship: ScaffoldedRelationshipRecord) -> None:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_SCAFFOLDED_RELATIONSHIP,
                    {
                        "id": relationship.id,
                        "source_id": relationship.source_id,
                        "target_id": relationship.target_id,
                        "rel_type": relationship.rel_type,
                        "properties": json.dumps(relationship.properties),
                    },
                )
            conn.commit()
        except Exception:
            pass

    def find_node_by_id(self, node_id: str) -> Optional[ScaffoldedFileNode]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(FLOW_GET_SCAFFOLDED_NODE_BY_ID, {"id": node_id})
                row = cur.fetchone()
                if row:
                    return ScaffoldedFileNode(
                        id=row[0],
                        file_path=row[1],
                        role=row[2],
                        layer=row[3],
                        feature_name=row[4],
                        package_name=row[5],
                        size_bytes=row[6],
                    )
        except Exception:
            pass
        return None

"""
================================================================================
BLUEPRINT: FILE SEARCH REPOSITORY ADAPTER
================================================================================
1. OVERVIEW:
   Concrete database repository adapter for AlloyDB/PostgreSQL implementing
   FileSearchRepositoryPort. Stores search telemetry and execution traces.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Database connection details and exception handling are documented here.
================================================================================
"""
from typing import Optional, Dict, Any
from src.features.file_search.infrastructure.ports.file_search_repository_port import FileSearchRepositoryPort
from src.features.file_search.infrastructure.queries.file_search_queries import (
    FLOW_INSERT_FILE_SEARCH_RECORD,
    FLOW_GET_FILE_SEARCH_RECORD_BY_ID,
)


class AlloyDBFileSearchRepository(FileSearchRepositoryPort):
    def __init__(self, db_url: str) -> None:
        self.db_url = db_url.replace("@localhost:", "@127.0.0.1:")
        self._conn = None
        self._memory_fallback: Dict[str, Dict[str, Any]] = {}

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

    def record_search(self, record_id: str, root_path: str, total_found: int, duration_ms: float) -> None:
        self._memory_fallback[record_id] = {
            "id": record_id,
            "root_path": root_path,
            "total_found": total_found,
            "execution_ms": duration_ms,
        }
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_FILE_SEARCH_RECORD,
                    {
                        "id": record_id,
                        "root_path": root_path,
                        "total_found": total_found,
                        "execution_ms": duration_ms,
                    },
                )
            conn.commit()
        except Exception:
            pass

    def get_search_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(FLOW_GET_FILE_SEARCH_RECORD_BY_ID, {"id": record_id})
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "root_path": row[1],
                        "total_found": row[2],
                        "execution_ms": row[3],
                    }
        except Exception:
            pass
        return self._memory_fallback.get(record_id)

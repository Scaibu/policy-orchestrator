"""
================================================================================
BLUEPRINT: CODE SEARCH REPOSITORY ADAPTER
================================================================================
1. OVERVIEW:
   Concrete database repository adapter for AlloyDB/PostgreSQL implementing
   CodeSearchRepositoryPort. Stores search metrics and provides in-memory fallback.
================================================================================
"""
from typing import Optional, Dict, Any
from src.features.code_search.infrastructure.ports.code_search_repository_port import CodeSearchRepositoryPort
from src.features.code_search.infrastructure.queries.code_search_queries import (
    FLOW_INSERT_CODE_SEARCH_RECORD,
    FLOW_GET_CODE_SEARCH_RECORD_BY_ID,
)


class AlloyDBCodeSearchRepository(CodeSearchRepositoryPort):
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

    def record_search(self, record_id: str, pattern_count: int, matches_found: int, duration_ms: float) -> None:
        self._memory_fallback[record_id] = {
            "id": record_id,
            "pattern_count": pattern_count,
            "matches_found": matches_found,
            "execution_ms": duration_ms,
        }
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_CODE_SEARCH_RECORD,
                    {
                        "id": record_id,
                        "pattern_count": pattern_count,
                        "matches_found": matches_found,
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
                cur.execute(FLOW_GET_CODE_SEARCH_RECORD_BY_ID, {"id": record_id})
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "pattern_count": row[1],
                        "matches_found": row[2],
                        "execution_ms": row[3],
                    }
        except Exception:
            pass
        return self._memory_fallback.get(record_id)

"""
================================================================================
BLUEPRINT: MODULE RESOLVER REPOSITORY ADAPTER
================================================================================
1. OVERVIEW:
   Concrete database repository adapter for AlloyDB/PostgreSQL implementing
   ModuleResolverRepositoryPort. Stores resolution telemetry with memory fallback.
================================================================================
"""
from typing import Optional, Dict, Any
from src.features.module_resolver.infrastructure.ports.module_resolver_repository_port import ModuleResolverRepositoryPort
from src.features.module_resolver.infrastructure.queries.module_resolver_queries import (
    FLOW_INSERT_MODULE_RESOLUTION_RECORD,
    FLOW_GET_MODULE_RESOLUTION_RECORD_BY_ID,
)


class AlloyDBModuleResolverRepository(ModuleResolverRepositoryPort):
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

    def record_resolution(
        self, record_id: str, module_name: str, is_valid: bool, has_collision: bool, duration_ms: float
    ) -> None:
        self._memory_fallback[record_id] = {
            "id": record_id,
            "module_name": module_name,
            "is_valid": is_valid,
            "collision_detected": has_collision,
            "execution_ms": duration_ms,
        }
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_MODULE_RESOLUTION_RECORD,
                    {
                        "id": record_id,
                        "module_name": module_name,
                        "is_valid": is_valid,
                        "collision_detected": has_collision,
                        "execution_ms": duration_ms,
                    },
                )
            conn.commit()
        except Exception:
            pass

    def get_resolution_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(FLOW_GET_MODULE_RESOLUTION_RECORD_BY_ID, {"id": record_id})
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "module_name": row[1],
                        "is_valid": row[2],
                        "collision_detected": row[3],
                        "execution_ms": row[4],
                    }
        except Exception:
            pass
        return self._memory_fallback.get(record_id)

"""
================================================================================
BLUEPRINT: GIT VERIFIER REPOSITORY ADAPTER
================================================================================
1. OVERVIEW:
   Concrete database repository adapter for AlloyDB/PostgreSQL implementing
   GitVerifierRepositoryPort. Stores Git verification records with memory fallback.
================================================================================
"""
from typing import Optional, Dict, Any
from src.features.git_verifier.infrastructure.ports.git_verifier_repository_port import GitVerifierRepositoryPort
from src.features.git_verifier.infrastructure.queries.git_verifier_queries import (
    FLOW_INSERT_GIT_VERIFICATION_RECORD,
    FLOW_GET_GIT_VERIFICATION_RECORD_BY_ID,
)


class AlloyDBGitVerifierRepository(GitVerifierRepositoryPort):
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

    def record_verification(
        self, record_id: str, repo_path: str, is_clean: bool, submodules_count: int, duration_ms: float
    ) -> None:
        self._memory_fallback[record_id] = {
            "id": record_id,
            "repo_path": repo_path,
            "is_clean": is_clean,
            "submodules_count": submodules_count,
            "execution_ms": duration_ms,
        }
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_GIT_VERIFICATION_RECORD,
                    {
                        "id": record_id,
                        "repo_path": repo_path,
                        "is_clean": is_clean,
                        "submodules_count": submodules_count,
                        "execution_ms": duration_ms,
                    },
                )
            conn.commit()
        except Exception:
            pass

    def get_verification_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(FLOW_GET_GIT_VERIFICATION_RECORD_BY_ID, {"id": record_id})
                row = cur.fetchone()
                if row:
                    return {
                        "id": row[0],
                        "repo_path": row[1],
                        "is_clean": row[2],
                        "submodules_count": row[3],
                        "execution_ms": row[4],
                    }
        except Exception:
            pass
        return self._memory_fallback.get(record_id)

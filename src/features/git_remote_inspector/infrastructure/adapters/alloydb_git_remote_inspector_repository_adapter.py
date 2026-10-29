from typing import Optional, Dict, Any
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_repository_port import (
    GitRemoteInspectorRepositoryPort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    DatabaseConnectionError,
    DatabaseConstraintViolationError,
    DatabaseTimeoutError,
)
from src.features.git_remote_inspector.infrastructure.queries.git_remote_inspector_queries import (
    FLOW_INSERT_REMOTE_INSPECTION_RECORD,
    FLOW_SELECT_REMOTE_INSPECTION_RECORD_BY_ID,
)


class AlloyDBGitRemoteInspectorRepositoryAdapter(GitRemoteInspectorRepositoryPort):
    def __init__(self, db_url: str) -> None:
        self._db_url: str = db_url.replace("@localhost:", "@127.0.0.1:")
        self._conn = None
        self._memory_fallback: Dict[str, Dict[str, Any]] = {}

    def _map_db_exception(self, exc: Exception) -> Exception:
        import psycopg2
        if isinstance(exc, psycopg2.IntegrityError):
            return DatabaseConstraintViolationError(str(exc))
        if isinstance(exc, psycopg2.OperationalError):
            msg = str(exc)
            if "timeout" in msg.lower():
                return DatabaseTimeoutError(msg)
            return DatabaseConnectionError(msg)
        return exc

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
        self._conn = psycopg2.connect(self._db_url)
        return self._conn

    def save_inspection_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        record_id = record.get("id", "")
        self._memory_fallback[record_id] = dict(record)
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    FLOW_INSERT_REMOTE_INSPECTION_RECORD,
                    {
                        "id": record.get("id"),
                        "repo_url": record.get("repo_url"),
                        "ref_name": record.get("ref_name"),
                        "path_pattern": record.get("path_pattern"),
                        "total_found": record.get("total_found", 0),
                        "execution_ms": record.get("execution_ms", 0.0),
                    },
                )
                conn.commit()
            return Result.ok(True)
        except Exception as exc:
            mapped = self._map_db_exception(exc)
            return Result.err(mapped)

    def get_inspection_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        if record_id in self._memory_fallback:
            return Result.ok(self._memory_fallback[record_id])
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(FLOW_SELECT_REMOTE_INSPECTION_RECORD_BY_ID, {"id": record_id})
                row = cur.fetchone()
                if row:
                    return Result.ok({
                        "id": row[0],
                        "repo_url": row[1],
                        "ref_name": row[2],
                        "path_pattern": row[3],
                        "total_found": row[4],
                        "execution_ms": row[5],
                        "created_at": str(row[6]),
                    })
            return Result.ok(None)
        except Exception as exc:
            mapped = self._map_db_exception(exc)
            return Result.err(mapped)

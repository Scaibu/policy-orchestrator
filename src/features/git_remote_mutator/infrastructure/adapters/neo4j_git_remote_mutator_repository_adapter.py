from typing import Optional, Dict, Any
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutator_repository_port import (
    GitRemoteMutatorRepositoryPort,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    DatabaseConnectionError,
    DatabaseConstraintViolationError,
    DatabaseTimeoutError,
)
from src.features.git_remote_mutator.infrastructure.queries.git_remote_mutator_cypher_queries import (
    CYPHER_UPDATE_BRANCH_POINTER,
    CYPHER_INSERT_MUTATION_LEDGER,
    CYPHER_GET_MUTATION_LEDGER,
)


class Neo4jGitRemoteMutatorRepositoryAdapter(GitRemoteMutatorRepositoryPort):
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        graph_client: Optional[Neo4jGraphAdapter] = None,
    ) -> None:
        if graph_client is not None:
            self._graph_client = graph_client
        else:
            from src.infra.config.config_loader import AppConfigLoader
            cfg = AppConfigLoader.get_knowledge_graph_config()
            resolved_uri = uri or cfg.get("uri")
            resolved_user = user or cfg.get("user")
            resolved_pw = password or cfg.get("password")
            resolved_db = database or cfg.get("database", "neo4j")
            self._graph_client = Neo4jGraphAdapter(
                uri=resolved_uri,
                user=resolved_user,
                password=resolved_pw,
                database=resolved_db,
            )
        self._memory_fallback: Dict[str, Dict[str, Any]] = {}

    def _map_graph_exception(self, exc: Exception) -> Exception:
        msg = str(exc).lower()
        if "timeout" in msg:
            return DatabaseTimeoutError(str(exc))
        if "constraint" in msg or "already exists" in msg:
            return DatabaseConstraintViolationError(str(exc))
        return DatabaseConnectionError(str(exc))

    def save_mutation_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        record_id = str(record.get("id", ""))
        self._memory_fallback[record_id] = dict(record)
        try:
            params = {
                "id": record_id,
                "repo_url": str(record.get("repo_url", "")),
                "target_branch": str(record.get("target_branch", "")),
                "parent_commit_sha": str(record.get("parent_commit_sha", "")),
                "new_commit_sha": str(record.get("new_commit_sha", "")),
                "files_count": int(record.get("files_count", 1)),
                "commit_message": str(record.get("commit_message", "")),
                "execution_ms": float(record.get("execution_ms", 0.0)),
            }
            self._graph_client.query_cypher(CYPHER_INSERT_MUTATION_LEDGER, params)
            return Result.ok(True)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

    def get_mutation_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        if record_id in self._memory_fallback:
            return Result.ok(self._memory_fallback[record_id])
        try:
            res = self._graph_client.query_cypher(
                CYPHER_GET_MUTATION_LEDGER, {"id": record_id}
            )
            if not res.records:
                return Result.ok(None)
            row = res.records[0]
            record = {
                "id": row.get("id"),
                "repo_url": row.get("repo_url"),
                "target_branch": row.get("target_branch"),
                "parent_commit_sha": row.get("parent_commit_sha"),
                "new_commit_sha": row.get("new_commit_sha"),
                "files_count": row.get("files_count"),
                "commit_message": row.get("commit_message"),
                "execution_ms": row.get("execution_ms"),
                "created_at": str(row.get("created_at") or ""),
            }
            return Result.ok(record)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

    def update_branch_pointer(
        self,
        ctx: TraceContext,
        repo_url: str,
        branch: str,
        parent_sha: str,
        new_commit_sha: str,
        commit_message: str,
    ) -> Result[bool, Exception]:
        try:
            params = {
                "repo_url": repo_url,
                "branch": branch,
                "parent_sha": parent_sha,
                "new_commit_sha": new_commit_sha,
                "commit_message": commit_message,
            }
            self._graph_client.query_cypher(CYPHER_UPDATE_BRANCH_POINTER, params)
            return Result.ok(True)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

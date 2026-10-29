"""
Neo4j Graph Adapter for Git Remote Inspector.
Implements GitRemoteInspectorRepositoryPort using Cypher transactional endpoints
with normalized first-class graph entities for Merkle caching and lineage tracking.
"""

import json
from typing import Optional, Dict, Any, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_repository_port import (
    GitRemoteInspectorRepositoryPort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    DatabaseConnectionError,
    DatabaseConstraintViolationError,
    DatabaseTimeoutError,
)
from src.features.git_remote_inspector.infrastructure.queries.git_remote_inspector_cypher_queries import (
    CYPHER_MERGE_IMMUTABLE_GRAPH,
    CYPHER_RECONSTRUCT_CACHED_ENDPOINT,
    CYPHER_INSERT_INSPECTION_RECORD,
    CYPHER_GET_INSPECTION_RECORD,
)


class Neo4jGitRemoteInspectorRepositoryAdapter(GitRemoteInspectorRepositoryPort):
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
        self._cache_fallback: Dict[str, Dict[str, Any]] = {}

    def _map_graph_exception(self, exc: Exception) -> Exception:
        msg = str(exc).lower()
        if "timeout" in msg:
            return DatabaseTimeoutError(str(exc))
        if "constraint" in msg or "already exists" in msg:
            return DatabaseConstraintViolationError(str(exc))
        return DatabaseConnectionError(str(exc))

    def save_inspection_record(
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
                "ref_name": str(record.get("ref_name", "")),
                "path_pattern": str(record.get("path_pattern") or ""),
                "total_found": int(record.get("total_found", 0)),
                "execution_ms": float(record.get("execution_ms", 0.0)),
            }
            self._graph_client.query_cypher(CYPHER_INSERT_INSPECTION_RECORD, params)
            return Result.ok(True)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

    def get_inspection_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        if record_id in self._memory_fallback:
            return Result.ok(self._memory_fallback[record_id])
        try:
            res = self._graph_client.query_cypher(
                CYPHER_GET_INSPECTION_RECORD, {"id": record_id}
            )
            if not res.records:
                return Result.ok(None)
            row = res.records[0]
            record = {
                "id": row.get("id"),
                "repo_url": row.get("repo_url"),
                "ref_name": row.get("ref_name"),
                "path_pattern": row.get("path_pattern"),
                "total_found": row.get("total_found"),
                "execution_ms": row.get("execution_ms"),
                "created_at": str(row.get("created_at") or ""),
            }
            return Result.ok(record)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

    def find_cached_endpoint(
        self,
        ctx: TraceContext,
        repo_url: str,
        commit_sha: str,
        endpoint_or_entity: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        mem_key = f"{repo_url}:{commit_sha}:{endpoint_or_entity}"
        if mem_key in self._cache_fallback:
            return Result.ok(self._cache_fallback[mem_key])
        try:
            params = {
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "target": endpoint_or_entity,
            }
            res = self._graph_client.query_cypher(CYPHER_RECONSTRUCT_CACHED_ENDPOINT, params)
            if not res.records:
                return Result.ok(None)
            row = res.records[0]
            if not row.get("target_endpoint_or_entity"):
                return Result.ok(None)

            req_body = row.get("request_body_schema")
            resp_200 = row.get("response_200_schema")
            sample_payload = row.get("sample_request_payload")

            signature = {
                "protocol": row.get("protocol") or "HTTP_REST",
                "http_method": row.get("http_method") or "POST",
                "path": row.get("path") or endpoint_or_entity,
                "summary": row.get("summary") or None,
                "auth_scheme": row.get("auth_scheme") or None,
                "is_deprecated": bool(row.get("is_deprecated", False)),
                "path_params": [],
                "query_params": [],
                "request_headers": [],
                "request_body_schema": json.loads(req_body) if isinstance(req_body, str) else (req_body or {}),
                "sample_request_payload": json.loads(sample_payload) if isinstance(sample_payload, str) else (sample_payload or {}),
                "response_200_schema": json.loads(resp_200) if isinstance(resp_200, str) else (resp_200 or {}),
                "sample_curl": row.get("sample_curl") or "",
                "client_code_snippet": row.get("client_code_snippet") or "",
            }

            dto_list = row.get("related_dto_files") or []
            if not isinstance(dto_list, list):
                dto_list = [dto_list]

            result_data = {
                "repo_url": row.get("repo_url") or repo_url,
                "provider": "github",
                "ref_name": commit_sha,
                "target_endpoint_or_entity": endpoint_or_entity,
                "match_type": row.get("match_type") or "OPENAPI_SPEC",
                "contract_file": row.get("contract_file") or None,
                "signature": signature,
                "related_dto_files": dto_list,
                "call_chain": [],
                "cached": True,
            }
            self._cache_fallback[mem_key] = result_data
            return Result.ok(result_data)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

    def save_endpoint_relationship(
        self,
        ctx: TraceContext,
        repo_url: str,
        commit_sha: str,
        ref_name: str,
        result_data: Dict[str, Any],
    ) -> Result[bool, Exception]:
        target = str(result_data.get("target_endpoint_or_entity", ""))
        mem_key = f"{repo_url}:{commit_sha}:{target}"
        self._cache_fallback[mem_key] = dict(result_data)
        try:
            sig = result_data.get("signature", {})
            if hasattr(sig, "model_dump"):
                sig = sig.model_dump()
            elif not isinstance(sig, dict):
                sig = {}

            req_body_json = json.dumps(sig.get("request_body_schema", {}))
            sample_payload_json = json.dumps(sig.get("sample_request_payload", {}))
            resp_200_json = json.dumps(sig.get("response_200_schema", {}))

            params = {
                "repo_url": repo_url,
                "commit_sha": commit_sha,
                "ref_name": ref_name,
                "target": target,
                "http_method": str(sig.get("http_method", "POST")),
                "path": str(sig.get("path") or target),
                "protocol": str(sig.get("protocol", "HTTP_REST")),
                "summary": str(sig.get("summary") or ""),
                "auth_scheme": str(sig.get("auth_scheme") or ""),
                "is_deprecated": bool(sig.get("is_deprecated", False)),
                "sample_curl": str(sig.get("sample_curl") or ""),
                "client_code_snippet": str(sig.get("client_code_snippet") or ""),
                "request_body_schema": req_body_json,
                "sample_request_payload": sample_payload_json,
                "response_200_schema": resp_200_json,
                "match_type": str(result_data.get("match_type", "OPENAPI_SPEC")),
                "contract_file": str(result_data.get("contract_file") or ""),
                "related_dto_files": result_data.get("related_dto_files", []),
            }
            self._graph_client.query_cypher(CYPHER_MERGE_IMMUTABLE_GRAPH, params)
            return Result.ok(True)
        except Exception as exc:
            mapped = self._map_graph_exception(exc)
            return Result.err(mapped)

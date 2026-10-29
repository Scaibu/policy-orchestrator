import time
import logging
from typing import Dict, Any, Tuple
from fastapi import Request
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.http_constants import (
    HEADER_X_TRACE_ID,
    HEADER_TRACEPARENT,
    HEADER_X_REQUEST_ID,
    HEADER_X_CORRELATION_ID,
    HEADER_X_CAUSATION_ID,
    HEADER_X_TENANT_ID,
    HEADER_X_CLIENT_ID,
)
from src.features.git_remote_inspector.service.git_remote_inspector_service import (
    GitRemoteInspectorDomainService,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_schema import (
    GitRemoteInspectorSchema,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    GetTreeHierarchyCommand,
    ListBranchesCommand,
    DiscoverContractsCommand,
    TraceCallChainCommand,
    GitRemoteBaseError,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope

logger = logging.getLogger("git_remote_inspector.handler")


class GitRemoteInspectorRestHandler:
    def __init__(self, service: GitRemoteInspectorDomainService) -> None:
        self.service = service

    def _extract_trace_context(self, req: Request) -> Tuple[TraceContext, str, str, str]:
        trace_id = req.headers.get(HEADER_X_TRACE_ID) or req.headers.get(HEADER_TRACEPARENT) or getattr(req.state, "trace_id", "synthetic-trace-id")
        request_id = req.headers.get(HEADER_X_REQUEST_ID, f"req-{trace_id[:8]}")
        correlation_id = req.headers.get(HEADER_X_CORRELATION_ID, f"corr-{trace_id[:12]}")
        causation_id = req.headers.get(HEADER_X_CAUSATION_ID)
        tenant_id = req.headers.get(HEADER_X_TENANT_ID)
        client_id = req.headers.get(HEADER_X_CLIENT_ID)
        ctx = TraceContext(
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            tenant_id=tenant_id,
            user_id=client_id,
        )
        return ctx, trace_id, request_id, correlation_id

    def _map_error_to_envelope(
        self,
        exc: Exception,
        fallback_code: str,
        trace_id: str,
        request_id: str,
        correlation_id: str,
        execution_time_ms: int,
    ) -> Dict[str, Any]:
        if isinstance(exc, GitRemoteBaseError):
            code = exc.canonical_code
            status_code = exc.status_code
            message = exc.message
            retryable = exc.retryable
            details = exc.details
        else:
            code = fallback_code
            status_code = 500
            message = str(exc)
            retryable = False
            details = None

        logger.error(
            f"Git Remote Inspector error: code={code} status={status_code} msg={message} trace_id={trace_id}"
        )
        return build_error_envelope(
            code=code,
            message=message,
            status_code=status_code,
            details=details,
            retryable=retryable,
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms,
        )

    def handle_search_files(self, req: Request, payload: SearchFilesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Searching remote files: repo={payload.repo_url} ref={payload.ref_name} pattern={payload.path_pattern} trace_id={trace_id}")
        result = self.service.search_files(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_search_files(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_SEARCH_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_get_file_content(self, req: Request, payload: GetFileContentCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Getting remote file content: repo={payload.repo_url} path={payload.file_path} ref={payload.ref_name} trace_id={trace_id}")
        result = self.service.get_file_content(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_get_file_content(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_CONTENT_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_get_diff(self, req: Request, payload: GetDiffCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Getting remote diff: repo={payload.repo_url} base={payload.base_ref} head={payload.head_ref} trace_id={trace_id}")
        result = self.service.get_diff(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_get_diff(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_DIFF_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_get_tree(self, req: Request, payload: GetTreeHierarchyCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Getting remote tree hierarchy: repo={payload.repo_url} ref={payload.ref_name} max_depth={payload.max_depth} trace_id={trace_id}")
        result = self.service.get_tree_hierarchy(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_get_tree(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_TREE_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_list_branches(self, req: Request, payload: ListBranchesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Listing remote branches: repo={payload.repo_url} trace_id={trace_id}")
        result = self.service.list_branches(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_list_branches(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_BRANCHES_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_discover_contracts(self, req: Request, payload: DiscoverContractsCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Discovering remote contracts: repo={payload.repo_url} ref={payload.ref_name} trace_id={trace_id}")
        result = self.service.discover_contracts(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_discover_contracts(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_DISCOVERY_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_trace_call_chain(self, req: Request, payload: TraceCallChainCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Tracing call chain: repo={payload.repo_url} target={payload.endpoint_or_symbol} depth={payload.max_depth} trace_id={trace_id}")
        result = self.service.trace_call_chain(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_trace_call_chain(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            exc=result.error(),
            fallback_code="GIT_REMOTE_CALL_CHAIN_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

import time
from typing import Dict, Any, Tuple
from fastapi import Request
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.http_constants import (
    HEADER_X_TRACE_ID,
    HEADER_TRACEPARENT,
    HEADER_X_REQUEST_ID,
    HEADER_X_CORRELATION_ID,
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
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class GitRemoteInspectorRestHandler:
    def __init__(self, service: GitRemoteInspectorDomainService) -> None:
        self.service = service

    def _extract_trace_context(self, req: Request) -> Tuple[TraceContext, str, str, str]:
        trace_id = req.headers.get(HEADER_X_TRACE_ID) or req.headers.get(HEADER_TRACEPARENT) or "synthetic-trace-id"
        request_id = req.headers.get(HEADER_X_REQUEST_ID, "req-default")
        correlation_id = req.headers.get(HEADER_X_CORRELATION_ID, "corr-default")
        ctx = TraceContext(
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
        )
        return ctx, trace_id, request_id, correlation_id

    def handle_search_files(self, req: Request, payload: SearchFilesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.search_files(ctx, payload)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_search_files(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="GIT_REMOTE_SEARCH_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_get_file_content(self, req: Request, payload: GetFileContentCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.get_file_content(ctx, payload)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_get_file_content(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="GIT_REMOTE_CONTENT_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_get_diff(self, req: Request, payload: GetDiffCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.get_diff(ctx, payload)
        if result.is_ok:
            data = GitRemoteInspectorSchema.to_api_get_diff(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="GIT_REMOTE_DIFF_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

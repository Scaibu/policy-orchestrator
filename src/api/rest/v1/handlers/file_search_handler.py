"""
================================================================================
BLUEPRINT: FILE SEARCH INGRESS DELIVERY HANDLER
================================================================================
1. OVERVIEW:
   Handles REST API ingress for directory tree searching, extracts W3C trace
   headers, manages error boundaries, and serializes RFC envelopes.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Workflow logic is fully captured in this top header block.
================================================================================
"""
import time
from typing import Dict, Any
from fastapi import Request
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.http_constants import (
    HEADER_X_TRACE_ID,
    HEADER_TRACEPARENT,
    HEADER_X_REQUEST_ID,
    HEADER_X_CORRELATION_ID,
)
from src.features.file_search.service.file_search_service import FileSearchDomainService
from src.features.file_search.contracts.file_search_schema import FileSearchSchema
from src.features.file_search.contracts.file_search_types import FileSearchCommand
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class FileSearchRestHandler:
    def __init__(self, service: FileSearchDomainService) -> None:
        self.service = service

    def _extract_trace_context(self, req: Request) -> tuple[TraceContext, str, str, str]:
        trace_id = req.headers.get(HEADER_X_TRACE_ID) or req.headers.get(HEADER_TRACEPARENT) or "synthetic-trace-id"
        request_id = req.headers.get(HEADER_X_REQUEST_ID, "req-default")
        correlation_id = req.headers.get(HEADER_X_CORRELATION_ID, "corr-default")
        ctx = TraceContext(
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
        )
        return ctx, trace_id, request_id, correlation_id

    def handle_search_files(self, req: Request, payload: FileSearchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.search_files(ctx, payload)
        if result.is_ok:
            data = FileSearchSchema.to_api_search_files(result.unwrap())
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
                code="FILE_SEARCH_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

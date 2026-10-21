"""
================================================================================
BLUEPRINT: CODE SEARCH INGRESS DELIVERY HANDLER
================================================================================
1. OVERVIEW:
   Handles REST API ingress for pattern and text matching across files,
   extracts trace context, and serializes RFC response envelopes.
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
from src.features.code_search.service.code_search_service import CodeSearchDomainService
from src.features.code_search.contracts.code_search_schema import CodeSearchSchema
from src.features.code_search.contracts.code_search_types import CodeSearchCommand
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class CodeSearchRestHandler:
    def __init__(self, service: CodeSearchDomainService) -> None:
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

    def handle_search_code(self, req: Request, payload: CodeSearchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.search_code(ctx, payload)
        if result.is_ok:
            data = CodeSearchSchema.to_api_search_code(result.unwrap())
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
                code="CODE_SEARCH_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

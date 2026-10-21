"""
================================================================================
BLUEPRINT: MODULE RESOLVER INGRESS DELIVERY HANDLER
================================================================================
1. OVERVIEW:
   Handles REST API ingress for Python module import resolution, extracts trace
   context, and serializes RFC response envelopes.
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
from src.features.module_resolver.service.module_resolver_service import ModuleResolverDomainService
from src.features.module_resolver.contracts.module_resolver_schema import ModuleResolverSchema
from src.features.module_resolver.contracts.module_resolver_types import ResolveModuleCommand
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class ModuleResolverRestHandler:
    def __init__(self, service: ModuleResolverDomainService) -> None:
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

    def handle_resolve_module(self, req: Request, payload: ResolveModuleCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.resolve_module(ctx, payload)
        if result.is_ok:
            data = ModuleResolverSchema.to_api_resolve_module(result.unwrap())
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
                code="MODULE_RESOLVE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

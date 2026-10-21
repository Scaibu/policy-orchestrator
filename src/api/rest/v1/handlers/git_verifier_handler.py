"""
================================================================================
BLUEPRINT: GIT VERIFIER INGRESS DELIVERY HANDLER
================================================================================
1. OVERVIEW:
   Handles REST API ingress for Git Merkle tree and submodule verification,
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
from src.features.git_verifier.service.git_verifier_service import GitVerifierDomainService
from src.features.git_verifier.contracts.git_verifier_schema import GitVerifierSchema
from src.features.git_verifier.contracts.git_verifier_types import VerifyGitCommand
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class GitVerifierRestHandler:
    def __init__(self, service: GitVerifierDomainService) -> None:
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

    def handle_verify_git(self, req: Request, payload: VerifyGitCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.verify_git(ctx, payload)
        if result.is_ok:
            data = GitVerifierSchema.to_api_verify_git(result.unwrap())
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
                code="GIT_VERIFY_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

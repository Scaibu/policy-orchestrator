"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER REST DELIVERY HANDLER
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone multi-repo Git submodule delivery handler:
     - Handles HTTP REST ingress for submodule add, update, pull, push, and sync-status.
     - Enforces trace context extraction (W3C traceparent, x-request-id, etc.).
     - Converts responses into standard RFC APIResponseEnvelope.
     - Sanitizes errors and maps to canonical error status envelopes.

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Closed-schema request handling.
================================================================================
"""

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
from src.features.git_submodule_manager.service.git_submodule_manager_service import (
    GitSubmoduleManagerDomainService,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_schema import (
    GitSubmoduleManagerSchema,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
    GitSubmoduleBaseError,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope

logger = logging.getLogger("git_submodule_manager.handler")


class GitSubmoduleManagerRestHandler:
    def __init__(self, service: GitSubmoduleManagerDomainService) -> None:
        self.service = service

    def _extract_trace_context(self, req: Request) -> Tuple[TraceContext, str, str, str]:
        trace_id = (
            req.headers.get(HEADER_X_TRACE_ID)
            or req.headers.get(HEADER_TRACEPARENT)
            or getattr(req.state, "trace_id", "synthetic-trace-id")
        )
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
        if isinstance(exc, GitSubmoduleBaseError):
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
            f"Git Submodule Manager error: code={code} status={status_code} msg={message} trace_id={trace_id}"
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

    def handle_add_submodule(self, req: Request, payload: AddSubmoduleCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.add_submodule(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitSubmoduleManagerSchema.to_api_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            result.error(), "SUBMODULE_ADD_FAILED", trace_id, request_id, correlation_id, exec_ms
        )

    def handle_update_submodule(self, req: Request, payload: UpdateSubmoduleCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.update_submodule(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitSubmoduleManagerSchema.to_api_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            result.error(), "SUBMODULE_UPDATE_FAILED", trace_id, request_id, correlation_id, exec_ms
        )

    def handle_pull_submodules(self, req: Request, payload: PullSubmodulesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.pull_submodules(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitSubmoduleManagerSchema.to_api_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            result.error(), "SUBMODULE_PULL_FAILED", trace_id, request_id, correlation_id, exec_ms
        )

    def handle_push_multi_repo(self, req: Request, payload: PushMultiRepoCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.push_multi_repo(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitSubmoduleManagerSchema.to_api_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            result.error(), "MULTI_REPO_PUSH_FAILED", trace_id, request_id, correlation_id, exec_ms
        )

    def handle_get_sync_status(self, req: Request, payload: SubmoduleSyncStatusCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.get_sync_status(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitSubmoduleManagerSchema.to_api_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_ms,
            )
        return self._map_error_to_envelope(
            result.error(), "SUBMODULE_SYNC_STATUS_FAILED", trace_id, request_id, correlation_id, exec_ms
        )

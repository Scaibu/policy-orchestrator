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
from src.features.git_branch_lifecycle.service.git_branch_lifecycle_service import (
    GitBranchLifecycleDomainService,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_schema import (
    GitBranchLifecycleSchema,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    GitBranchLifecycleBaseError,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope

logger = logging.getLogger("git_branch_lifecycle.handler")


class GitBranchLifecycleRestHandler:
    def __init__(self, service: GitBranchLifecycleDomainService) -> None:
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
        status_code = getattr(exc, "status_code", 500)
        code = getattr(exc, "canonical_code", fallback_code)
        retryable = getattr(exc, "retryable", False)
        details = getattr(exc, "details", None)
        message = str(exc)
        logger.error(
            f"Git Branch Lifecycle error: code={code} status={status_code} msg={message} trace_id={trace_id}"
        )
        return build_error_envelope(
            code=code,
            message=message,
            status_code=status_code,
            request_id=request_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            retryable=retryable,
            details=details,
            execution_time_ms=execution_time_ms,
        )

    async def handle_create_branch(self, req: Request, payload: CreateBranchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        try:
            res = self.service.create_branch(ctx, payload)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            if res.is_err:
                return self._map_error_to_envelope(
                    res.error(), "GIT_BRANCH_CREATE_FAILED", trace_id, request_id, correlation_id, execution_time_ms
                )
            data = GitBranchLifecycleSchema.to_api_result(res.unwrap())
            return build_success_envelope(
                data=data,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            return self._map_error_to_envelope(
                exc, "GIT_BRANCH_CREATE_UNHANDLED_ERROR", trace_id, request_id, correlation_id, execution_time_ms
            )

    async def handle_delete_branch(self, req: Request, payload: DeleteBranchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        try:
            res = self.service.delete_branch(ctx, payload)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            if res.is_err:
                return self._map_error_to_envelope(
                    res.error(), "GIT_BRANCH_DELETE_FAILED", trace_id, request_id, correlation_id, execution_time_ms
                )
            data = GitBranchLifecycleSchema.to_api_result(res.unwrap())
            return build_success_envelope(
                data=data,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            return self._map_error_to_envelope(
                exc, "GIT_BRANCH_DELETE_UNHANDLED_ERROR", trace_id, request_id, correlation_id, execution_time_ms
            )

    async def handle_merge_branch(self, req: Request, payload: MergeBranchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        try:
            res = self.service.merge_branch(ctx, payload)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            if res.is_err:
                return self._map_error_to_envelope(
                    res.error(), "GIT_BRANCH_MERGE_FAILED", trace_id, request_id, correlation_id, execution_time_ms
                )
            data = GitBranchLifecycleSchema.to_api_result(res.unwrap())
            return build_success_envelope(
                data=data,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            return self._map_error_to_envelope(
                exc, "GIT_BRANCH_MERGE_UNHANDLED_ERROR", trace_id, request_id, correlation_id, execution_time_ms
            )

    async def handle_create_repository(self, req: Request, payload: CreateRepositoryCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        try:
            res = self.service.create_repository(ctx, payload)
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            if res.is_err:
                return self._map_error_to_envelope(
                    res.error(), "GIT_REPO_CREATE_FAILED", trace_id, request_id, correlation_id, execution_time_ms
                )
            data = GitBranchLifecycleSchema.to_api_result(res.unwrap())
            return build_success_envelope(
                data=data,
                request_id=request_id,
                correlation_id=correlation_id,
                trace_id=trace_id,
                execution_time_ms=execution_time_ms,
            )
        except Exception as exc:
            execution_time_ms = int((time.perf_counter() - start) * 1000)
            return self._map_error_to_envelope(
                exc, "GIT_REPO_CREATE_UNHANDLED_ERROR", trace_id, request_id, correlation_id, execution_time_ms
            )

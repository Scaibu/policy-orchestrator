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
from src.features.git_remote_mutator.service.git_remote_mutator_service import (
    GitRemoteMutatorDomainService,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_schema import (
    GitRemoteMutatorSchema,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
    GitRemoteMutatorBaseError,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope

logger = logging.getLogger("git_remote_mutator.handler")


class GitRemoteMutatorRestHandler:
    def __init__(self, service: GitRemoteMutatorDomainService) -> None:
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
        if isinstance(exc, GitRemoteMutatorBaseError):
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
            f"Git Remote Mutator error: code={code} status={status_code} msg={message} trace_id={trace_id}"
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

    def handle_mutate_file(self, req: Request, payload: MutateFileCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Mutating remote file: repo={payload.repo_url} branch={payload.branch} path={payload.file_path} trace_id={trace_id}")
        result = self.service.mutate_file(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteMutatorSchema.to_api_mutation_result(result.unwrap())
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
            fallback_code="GIT_REMOTE_MUTATE_FILE_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_mutate_batch(self, req: Request, payload: MutateBatchCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Mutating remote batch: repo={payload.repo_url} branch={payload.branch} count={len(payload.mutations)} trace_id={trace_id}")
        result = self.service.mutate_batch(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteMutatorSchema.to_api_mutation_result(result.unwrap())
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
            fallback_code="GIT_REMOTE_MUTATE_BATCH_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

    def handle_delete_file(self, req: Request, payload: DeleteFileCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        logger.info(f"Deleting remote file: repo={payload.repo_url} branch={payload.branch} path={payload.file_path} trace_id={trace_id}")
        result = self.service.delete_file(ctx, payload)
        exec_ms = int((time.perf_counter() - start) * 1000)
        if result.is_ok:
            data = GitRemoteMutatorSchema.to_api_mutation_result(result.unwrap())
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
            fallback_code="GIT_REMOTE_DELETE_FILE_FAILED",
            trace_id=trace_id,
            request_id=request_id,
            correlation_id=correlation_id,
            execution_time_ms=exec_ms,
        )

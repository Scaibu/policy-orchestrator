import time
from typing import Dict, Any, Optional
from fastapi import Request
from src.shared.types.envelope_types import TraceContext
from src.shared.constants.http_constants import (
    HEADER_X_TRACE_ID,
    HEADER_TRACEPARENT,
    HEADER_X_REQUEST_ID,
    HEADER_X_CORRELATION_ID,
)
from src.features.folder_scaffold.service.folder_scaffold_service_port import FolderScaffoldServicePort
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service_port import FolderScaffoldAnalyzerServicePort
from src.features.folder_scaffold.contracts.folder_scaffold_schema import FolderScaffoldSchema
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldTreeCommand,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class FolderScaffoldRestHandler:
    def __init__(
        self,
        service: FolderScaffoldServicePort,
        analyzer: Optional[FolderScaffoldAnalyzerServicePort] = None,
    ) -> None:
        self.service = service
        self.analyzer = analyzer

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

    def handle_create_file(self, req: Request, payload: CreateFileCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.createFile(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_create_file(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="CREATE_FILE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_link_files(self, req: Request, payload: LinkFilesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.linkFiles(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_link_files(result.unwrap())
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
                code="LINK_FILES_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_scaffold_tree(self, req: Request, payload: ScaffoldTreeCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        result = self.service.scaffoldTree(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_scaffold_tree(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="SCAFFOLD_TREE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_analyze_impact(
        self,
        req: Request,
        target_id: str,
        direction: str = "upstream",
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        if not self.analyzer:
            return build_error_envelope(
                code="ANALYZER_UNAVAILABLE",
                message="Graph analyzer service is not configured",
                status_code=503,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        result = self.analyzer.analyze_impact(
            ctx=ctx,
            target_id=target_id,
            direction=direction,
            max_depth=max_depth,
        )
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_blast_radius(result.unwrap())
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
                code="ANALYZE_IMPACT_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_get_file_lineage(
        self,
        req: Request,
        file_path: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        ctx, trace_id, request_id, correlation_id = self._extract_trace_context(req)
        if not self.analyzer:
            return build_error_envelope(
                code="ANALYZER_UNAVAILABLE",
                message="Graph analyzer service is not configured",
                status_code=503,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        result = self.analyzer.get_file_lineage(
            ctx=ctx,
            file_path=file_path,
            max_depth=max_depth,
        )
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_file_lineage(result.unwrap())
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
                code="GET_LINEAGE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

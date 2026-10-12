import time
from typing import Dict, Any
from fastapi import Request
from src.shared.types.envelope_types import TraceContext
from src.features.folder_scaffold.service.folder_scaffold_service_port import FolderScaffoldServicePort
from src.features.folder_scaffold.schema.folder_scaffold_schema import FolderScaffoldSchema
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldFeatureCommand,
    ScaffoldPackageCommand,
)
from src.api.rest.envelope import build_success_envelope, build_error_envelope


class FolderScaffoldRestHandler:
    def __init__(self, service: FolderScaffoldServicePort) -> None:
        self.service = service

    def handle_create_file(self, req: Request, payload: CreateFileCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        ctx = TraceContext(trace_id=trace_id)
        result = self.service.createFile(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_create_file(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=201,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="CREATE_FILE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_link_files(self, req: Request, payload: LinkFilesCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        ctx = TraceContext(trace_id=trace_id)
        result = self.service.linkFiles(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_link_files(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=200,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="LINK_FILES_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_scaffold_feature(self, req: Request, payload: ScaffoldFeatureCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        ctx = TraceContext(trace_id=trace_id)
        result = self.service.scaffoldFeature(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_scaffold_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=201,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="SCAFFOLD_FEATURE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

    def handle_scaffold_package(self, req: Request, payload: ScaffoldPackageCommand) -> Dict[str, Any]:
        start = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        ctx = TraceContext(trace_id=trace_id)
        result = self.service.scaffoldPackage(ctx, payload)
        if result.is_ok:
            data = FolderScaffoldSchema.to_api_scaffold_result(result.unwrap())
            return build_success_envelope(
                data=data,
                status_code=201,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )
        else:
            return build_error_envelope(
                code="SCAFFOLD_PACKAGE_FAILED",
                message=str(result.error()),
                status_code=400,
                trace_id=trace_id,
                request_id=req.headers.get("x-request-id", "req-default"),
                correlation_id=req.headers.get("x-correlation-id", "corr-default"),
                execution_time_ms=int((time.perf_counter() - start) * 1000),
            )

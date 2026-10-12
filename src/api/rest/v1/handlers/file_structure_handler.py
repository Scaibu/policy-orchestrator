"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: FILE STRUCTURE REST DELIVERY HANDLER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Thin ingress REST handler for File Structure scaffolding and Impact Analysis APIs.
   Adheres 100% to RFC envelope specifications returning standardized
   `success`, `statusCode`, `data`, and `meta`.

2. ARCHITECTURAL LAYOUT & DESIGN PILLARS:
   - Zero-Inline-Comment Doctrine: All HTTP routing, status code mapping, and
     header extraction rules are articulated in this header.
   - Zero Domain Logic: Delegates all business decisions and impact calculation
     to FileStructureDomainService.
================================================================================
"""

import time
from typing import Dict, Any
from fastapi import Request
from src.api.rest.envelope import build_success_envelope, build_error_envelope
from src.features.file_structure.service.file_structure_service import FileStructureDomainService
from src.features.file_structure.schema.file_structure_schema import (
    AnalyzeImpactRequestSchema,
    ScanRepositoryRequestSchema,
    ScaffoldFeatureRequestSchema,
    ScaffoldPackageRequestSchema,
    LinkFilesRequestSchema,
    CreateFileRequestSchema,
    CreateScaffoldTemplateRequestSchema,
    UpdateScaffoldTemplateRequestSchema,
    FileStructureEntitySchema,
)

class FileStructureRestHandler:
    def __init__(self, service: FileStructureDomainService) -> None:
        self.service = service

    def handle_scaffold_feature(self, req: Request, payload: ScaffoldFeatureRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            scaffold_result = self.service.scaffold_feature(
                feature_name=payload.feature_name,
                base_dir=payload.base_dir,
                package_root=payload.package_root,
                with_router=payload.with_router,
                with_handler=payload.with_handler,
                port_type=payload.port_type,
            )
            api_data = FileStructureEntitySchema.to_api(scaffold_result)
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=api_data,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="FEATURE_SCAFFOLD_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_scaffold_package(self, req: Request, payload: ScaffoldPackageRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            scaffold_result = self.service.scaffold_package(
                package_name=payload.package_name,
                base_dir=payload.base_dir,
            )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=scaffold_result,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="PACKAGE_SCAFFOLD_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_analyze_impact(self, req: Request, payload: AnalyzeImpactRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            domain_result = self.service.analyze_node_impact(
                target_id=payload.target_id,
                direction=payload.direction,
                max_depth=payload.max_depth,
            )
            raw_dict = {
                "target_id": domain_result.target_id,
                "target_label": domain_result.target_label,
                "direction": domain_result.direction,
                "blast_radius_score": domain_result.blast_radius_score,
                "risk_level": domain_result.risk_level,
                "impacted_nodes": [
                    {
                        "id": n.id,
                        "label": n.label,
                        "path": n.path,
                        "depth": n.depth,
                        "relationship": n.relationship,
                        "riskFactor": n.risk_factor,
                    }
                    for n in domain_result.impacted_nodes
                ],
                "affected_layers": domain_result.affected_layers,
                "potential_breaking_risks": domain_result.potential_breaking_risks,
                "required_verification_commands": domain_result.required_verification_commands,
                "recommended_mitigation_recipe": domain_result.recommended_mitigation_recipe,
            }
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)

            return build_success_envelope(
                data=raw_dict,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="IMPACT_ANALYSIS_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_scan_repository(self, req: Request, payload: ScanRepositoryRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            sync_result = self.service.scan_and_sync_workspace(root_dir=payload.root_dir)
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=sync_result,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="REPOSITORY_SCAN_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_get_feature_map(self, req: Request, feature_name: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            mapping = self.service.get_feature_map(feature_name=feature_name)
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=mapping,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="FEATURE_MAP_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_link_files(self, req: Request, payload: LinkFilesRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            link_result = self.service.link_files(
                source=payload.source,
                rel_type=payload.rel_type,
                target=payload.target,
                properties=payload.properties,
            )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=link_result,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="LINK_FILES_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_create_file(self, req: Request, payload: CreateFileRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            create_result = self.service.create_file_with_relationship(
                file_path=payload.file_path,
                role=payload.role,
                rel_type=payload.rel_type,
                target_file_or_node=payload.target_file_or_node,
                direction=payload.direction,
                content=payload.content,
                package_root=payload.package_root,
                feature_name=payload.feature_name,
                properties=payload.properties,
            )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=create_result,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
                version="v1",
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="CREATE_FILE_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_list_templates(self, req: Request) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            templates = self.service.list_scaffold_templates()
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=templates,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="LIST_TEMPLATES_FAILED",
                message=str(exc),
                status_code=500,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_get_template(self, req: Request, template_type: str, version: str = "v1") -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            template = self.service.get_scaffold_template(template_type=template_type, version=version)
            if not template:
                exec_time_ms = int((time.perf_counter() - start_time) * 1000)
                return build_error_envelope(
                    code="TEMPLATE_NOT_FOUND",
                    message=f"Scaffold template '{template_type}' version '{version}' not found",
                    status_code=404,
                    trace_id=trace_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    execution_time_ms=exec_time_ms,
                )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=template,
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="GET_TEMPLATE_FAILED",
                message=str(exc),
                status_code=500,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_create_template(self, req: Request, payload: CreateScaffoldTemplateRequestSchema) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            created = self.service.create_scaffold_template(
                template_type=payload.template_type,
                version=payload.version,
                directories=payload.directories,
                files=payload.files,
                metadata=payload.metadata,
            )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data=created,
                status_code=201,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="CREATE_TEMPLATE_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_update_template(
        self,
        req: Request,
        template_id: str,
        payload: UpdateScaffoldTemplateRequestSchema,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            updated = self.service.update_scaffold_template(
                template_id=template_id,
                directories=payload.directories,
                files=payload.files,
                metadata=payload.metadata,
            )
            if not updated:
                exec_time_ms = int((time.perf_counter() - start_time) * 1000)
                return build_error_envelope(
                    code="TEMPLATE_NOT_FOUND",
                    message=f"Scaffold template '{template_id}' not found for update",
                    status_code=404,
                    trace_id=trace_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    execution_time_ms=exec_time_ms,
                )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data={"id": template_id, "updated": True},
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="UPDATE_TEMPLATE_FAILED",
                message=str(exc),
                status_code=400,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

    def handle_delete_template(self, req: Request, template_id: str) -> Dict[str, Any]:
        start_time = time.perf_counter()
        trace_id = req.headers.get("x-trace-id") or req.headers.get("traceparent") or "synthetic-trace-id"
        request_id = req.headers.get("x-request-id") or f"req-{int(time.time() * 1000)}"
        correlation_id = req.headers.get("x-correlation-id") or request_id

        try:
            deleted = self.service.delete_scaffold_template(template_id=template_id)
            if not deleted:
                exec_time_ms = int((time.perf_counter() - start_time) * 1000)
                return build_error_envelope(
                    code="TEMPLATE_NOT_FOUND",
                    message=f"Scaffold template '{template_id}' not found for deletion",
                    status_code=404,
                    trace_id=trace_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    execution_time_ms=exec_time_ms,
                )
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_success_envelope(
                data={"id": template_id, "deleted": True},
                status_code=200,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )
        except Exception as exc:
            exec_time_ms = int((time.perf_counter() - start_time) * 1000)
            return build_error_envelope(
                code="DELETE_TEMPLATE_FAILED",
                message=str(exc),
                status_code=500,
                trace_id=trace_id,
                request_id=request_id,
                correlation_id=correlation_id,
                execution_time_ms=exec_time_ms,
            )

KnowledgeGraphRestHandler = FileStructureRestHandler

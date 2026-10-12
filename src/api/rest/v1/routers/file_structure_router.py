"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: FILE STRUCTURE REST ROUTER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Dedicated FastAPI router mounting File Structure scaffolding, validation,
   and Knowledge Graph impact analysis endpoints.
   Adheres strictly to RFC standards and Guardrail 9 by loading all endpoint
   paths purely from package root config via AppConfigLoader.
   Zero inline comments within endpoint handlers.
================================================================================
"""

from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.api.rest.v1.handlers.file_structure_handler import FileStructureRestHandler
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
)
from src.api.rest.envelope import APIResponseEnvelope
from src.api.rest.v1.dependencies import get_graph_store

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("file_structure", "base") or "/file-structure",
    tags=["File Structure & Knowledge Graph"],
)

def get_file_structure_handler(graph_store=Depends(get_graph_store)) -> FileStructureRestHandler:
    service = FileStructureDomainService(graph_store=graph_store)
    return FileStructureRestHandler(service=service)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "scaffold_feature") or "/scaffold/feature",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def scaffold_feature_structure(
    payload: ScaffoldFeatureRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_scaffold_feature(req=req, payload=payload)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "scaffold_package") or "/scaffold/package",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def scaffold_package_structure(
    payload: ScaffoldPackageRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_scaffold_package(req=req, payload=payload)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "impact_analysis") or "/impact-analysis",
    response_model=APIResponseEnvelope,
)
def analyze_architectural_impact(
    payload: AnalyzeImpactRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_analyze_impact(req=req, payload=payload)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "scan_repository") or "/scan-repository",
    response_model=APIResponseEnvelope,
)
def scan_repository_workspace(
    payload: ScanRepositoryRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_scan_repository(req=req, payload=payload)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "link_files") or "/link-files",
    response_model=APIResponseEnvelope,
)
def link_architectural_files(
    payload: LinkFilesRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_link_files(req=req, payload=payload)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "create_file") or "/create-file",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def create_architectural_file(
    payload: CreateFileRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_create_file(req=req, payload=payload)

@router.get(
    AppConfigLoader.get_endpoint("file_structure", "feature_map") or "/feature/{feature_name}",
    response_model=APIResponseEnvelope,
)
def get_feature_architecture_map(
    feature_name: str,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_get_feature_map(req=req, feature_name=feature_name)

@router.get(
    AppConfigLoader.get_endpoint("file_structure", "templates_list") or "/scaffold/templates",
    response_model=APIResponseEnvelope,
)
def list_scaffold_templates(
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_list_templates(req=req)

@router.get(
    AppConfigLoader.get_endpoint("file_structure", "template_get") or "/scaffold/templates/{template_type}",
    response_model=APIResponseEnvelope,
)
def get_scaffold_template(
    template_type: str,
    req: Request,
    version: str = "v1",
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_get_template(req=req, template_type=template_type, version=version)

@router.post(
    AppConfigLoader.get_endpoint("file_structure", "template_create") or "/scaffold/templates",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def create_scaffold_template(
    payload: CreateScaffoldTemplateRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_create_template(req=req, payload=payload)

@router.put(
    AppConfigLoader.get_endpoint("file_structure", "template_update") or "/scaffold/templates/{template_id}",
    response_model=APIResponseEnvelope,
)
def update_scaffold_template(
    template_id: str,
    payload: UpdateScaffoldTemplateRequestSchema,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_update_template(req=req, template_id=template_id, payload=payload)

@router.delete(
    AppConfigLoader.get_endpoint("file_structure", "template_delete") or "/scaffold/templates/{template_id}",
    response_model=APIResponseEnvelope,
)
def delete_scaffold_template(
    template_id: str,
    req: Request,
    handler: FileStructureRestHandler = Depends(get_file_structure_handler),
):
    return handler.handle_delete_template(req=req, template_id=template_id)

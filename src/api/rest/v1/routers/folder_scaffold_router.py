from fastapi import APIRouter, Request, Depends, Query, status
from src.infra.config.config_loader import AppConfigLoader
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.features.folder_scaffold.infrastructure.adapters.folder_scaffold_repository_adapter import AlloyDBFolderScaffoldRepository
from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service import FolderScaffoldAnalyzerDomainService
from src.features.folder_scaffold.contracts.folder_scaffold_deps import FolderScaffoldServiceDeps
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldTreeCommand,
)
from src.api.rest.v1.handlers.folder_scaffold_handler import FolderScaffoldRestHandler
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("folder_scaffold", "base") or "/folder-scaffold",
    tags=["Folder & File Scaffold"],
)


def get_handler() -> FolderScaffoldRestHandler:
    db_url = AppConfigLoader.get_database_url()
    kg_cfg = AppConfigLoader.get_knowledge_graph_config()
    scaffold_cfg = AppConfigLoader.get_scaffold_config()
    repo = AlloyDBFolderScaffoldRepository(db_url=db_url)
    graph_store = Neo4jGraphAdapter(
        uri=kg_cfg.get("uri"),
        user=kg_cfg.get("user"),
        password=kg_cfg.get("password"),
        database=kg_cfg.get("database", "neo4j"),
    )
    deps = FolderScaffoldServiceDeps(
        repository=repo,
        graph_store=graph_store,
        scaffold_config=scaffold_cfg,
    )
    service = FolderScaffoldDomainService(deps=deps)
    analyzer = FolderScaffoldAnalyzerDomainService(repository=repo, graph_store=graph_store)
    return FolderScaffoldRestHandler(service=service, analyzer=analyzer)


@router.post(
    AppConfigLoader.get_endpoint("folder_scaffold", "create_file") or "/create-file",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def create_file_with_relationship(
    payload: CreateFileCommand,
    req: Request,
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_create_file(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("folder_scaffold", "link_files") or "/link-files",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def link_files(
    payload: LinkFilesCommand,
    req: Request,
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_link_files(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("folder_scaffold", "scaffold_tree") or "/scaffold-tree",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def scaffold_tree(
    payload: ScaffoldTreeCommand,
    req: Request,
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_scaffold_tree(req=req, payload=payload)


@router.get(
    AppConfigLoader.get_endpoint("folder_scaffold", "analyze_impact") or "/analyze/impact",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def analyze_impact(
    req: Request,
    target_id: str = Query(..., description="Target node ID to evaluate blast radius for"),
    direction: str = Query("upstream", description="Direction of impact ('upstream' or 'downstream')"),
    max_depth: int = Query(5, ge=1, le=20, description="Max traversal depth in the knowledge graph"),
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_analyze_impact(
        req=req,
        target_id=target_id,
        direction=direction,
        max_depth=max_depth,
    )


@router.get(
    AppConfigLoader.get_endpoint("folder_scaffold", "analyze_lineage") or "/analyze/lineage",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def analyze_lineage(
    req: Request,
    file_path: str = Query(..., description="Absolute or relative file path to trace"),
    max_depth: int = Query(5, ge=1, le=20, description="Max lineage traversal depth"),
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_get_file_lineage(
        req=req,
        file_path=file_path,
        max_depth=max_depth,
    )

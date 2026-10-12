from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.features.folder_scaffold.repository.folder_scaffold_repository import AlloyDBFolderScaffoldRepository
from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
from src.features.folder_scaffold.types.folder_scaffold_deps import FolderScaffoldServiceDeps
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldFeatureCommand,
    ScaffoldPackageCommand,
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
    return FolderScaffoldRestHandler(service=service)



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
    AppConfigLoader.get_endpoint("folder_scaffold", "scaffold_feature") or "/scaffold-feature",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def scaffold_feature(
    payload: ScaffoldFeatureCommand,
    req: Request,
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_scaffold_feature(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("folder_scaffold", "scaffold_package") or "/scaffold-package",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_201_CREATED,
)
def scaffold_package(
    payload: ScaffoldPackageCommand,
    req: Request,
    handler: FolderScaffoldRestHandler = Depends(get_handler),
):
    return handler.handle_scaffold_package(req=req, payload=payload)

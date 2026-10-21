"""
================================================================================
BLUEPRINT: FILE SEARCH INGRESS ROUTER
================================================================================
1. OVERVIEW:
   Mounts file search API endpoints, resolves dependencies, and routes requests
   to FileSearchRestHandler under W3C trace propagation.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Endpoints and wiring are defined in this top blueprint.
================================================================================
"""
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.file_search.contracts.file_search_types import FileSearchCommand
from src.features.file_search.service.file_search_service import FileSearchDomainService
from src.features.file_search.infrastructure.adapters.file_search_repository_adapter import AlloyDBFileSearchRepository
from src.api.rest.v1.handlers.file_search_handler import FileSearchRestHandler
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("file_search", "base") or "/file-search",
    tags=["File Search & Tree Traversal"],
)


def get_file_search_handler() -> FileSearchRestHandler:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBFileSearchRepository(db_url=db_url)
    service = FileSearchDomainService(repository=repo)
    return FileSearchRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("file_search", "scan") or "/scan",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def scan_directory(
    payload: FileSearchCommand,
    req: Request,
    handler: FileSearchRestHandler = Depends(get_file_search_handler),
):
    return handler.handle_search_files(req=req, payload=payload)

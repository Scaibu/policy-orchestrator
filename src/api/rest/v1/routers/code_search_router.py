"""
================================================================================
BLUEPRINT: CODE SEARCH INGRESS ROUTER
================================================================================
1. OVERVIEW:
   Mounts code pattern search API endpoints, resolves dependencies, and routes
   requests to CodeSearchRestHandler under W3C trace propagation.
================================================================================
"""
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.code_search.contracts.code_search_types import CodeSearchCommand
from src.features.code_search.service.code_search_service import CodeSearchDomainService
from src.features.code_search.infrastructure.adapters.code_search_repository_adapter import AlloyDBCodeSearchRepository
from src.api.rest.v1.handlers.code_search_handler import CodeSearchRestHandler
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("code_search", "base") or "/code-search",
    tags=["Code & Pattern Search"],
)


def get_code_search_handler() -> CodeSearchRestHandler:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBCodeSearchRepository(db_url=db_url)
    service = CodeSearchDomainService(repository=repo)
    return CodeSearchRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("code_search", "scan") or "/scan",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def scan_code_patterns(
    payload: CodeSearchCommand,
    req: Request,
    handler: CodeSearchRestHandler = Depends(get_code_search_handler),
):
    return handler.handle_search_code(req=req, payload=payload)

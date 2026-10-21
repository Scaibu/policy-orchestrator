"""
================================================================================
BLUEPRINT: MODULE RESOLVER INGRESS ROUTER
================================================================================
1. OVERVIEW:
   Mounts Python module resolution API endpoints, resolves dependencies, and
   routes requests to ModuleResolverRestHandler under W3C trace propagation.
================================================================================
"""
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.module_resolver.contracts.module_resolver_types import ResolveModuleCommand
from src.features.module_resolver.service.module_resolver_service import ModuleResolverDomainService
from src.features.module_resolver.infrastructure.adapters.module_resolver_repository_adapter import (
    AlloyDBModuleResolverRepository,
)
from src.api.rest.v1.handlers.module_resolver_handler import ModuleResolverRestHandler
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("module_resolver", "base") or "/module-resolver",
    tags=["Module Resolver (PEP 302 / 451)"],
)


def get_module_resolver_handler() -> ModuleResolverRestHandler:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBModuleResolverRepository(db_url=db_url)
    service = ModuleResolverDomainService(repository=repo)
    return ModuleResolverRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("module_resolver", "resolve") or "/resolve",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def resolve_module(
    payload: ResolveModuleCommand,
    req: Request,
    handler: ModuleResolverRestHandler = Depends(get_module_resolver_handler),
):
    return handler.handle_resolve_module(req=req, payload=payload)

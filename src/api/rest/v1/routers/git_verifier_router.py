"""
================================================================================
BLUEPRINT: GIT VERIFIER INGRESS ROUTER
================================================================================
1. OVERVIEW:
   Mounts Git repository verification API endpoints, resolves dependencies, and
   routes requests to GitVerifierRestHandler under W3C trace propagation.
================================================================================
"""
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_verifier.contracts.git_verifier_types import VerifyGitCommand
from src.features.git_verifier.service.git_verifier_service import GitVerifierDomainService
from src.features.git_verifier.infrastructure.adapters.git_verifier_repository_adapter import (
    AlloyDBGitVerifierRepository,
)
from src.api.rest.v1.handlers.git_verifier_handler import GitVerifierRestHandler
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("git_verifier", "base") or "/git-verifier",
    tags=["Git Verifier & Submodule Auditor"],
)


def get_git_verifier_handler() -> GitVerifierRestHandler:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBGitVerifierRepository(db_url=db_url)
    service = GitVerifierDomainService(repository=repo)
    return GitVerifierRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("git_verifier", "verify") or "/verify",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def verify_git_state(
    payload: VerifyGitCommand,
    req: Request,
    handler: GitVerifierRestHandler = Depends(get_git_verifier_handler),
):
    return handler.handle_verify_git(req=req, payload=payload)

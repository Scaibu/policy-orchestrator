import os
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_deps import (
    GitBranchLifecycleServiceDeps,
)
from src.features.git_branch_lifecycle.service.git_branch_lifecycle_service import (
    GitBranchLifecycleDomainService,
)
from src.features.git_branch_lifecycle.infrastructure.adapters.github_branch_lifecycle_client_adapter import (
    GitHubBranchLifecycleClientAdapter,
)
from src.features.git_branch_lifecycle.infrastructure.adapters.neo4j_branch_lifecycle_repository_adapter import (
    Neo4jBranchLifecycleRepositoryAdapter,
)
from src.features.git_branch_lifecycle.infrastructure.adapters.alloydb_branch_lifecycle_repository_adapter import (
    AlloyDBBranchLifecycleRepositoryAdapter,
)
from src.api.rest.v1.handlers.git_branch_lifecycle_handler import (
    GitBranchLifecycleRestHandler,
)
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("git_branch_lifecycle", "base") or "/git-branch-lifecycle",
    tags=["Git Branch Lifecycle (Zero-Clone Branch & Repo Management)"],
)


def get_git_branch_lifecycle_handler() -> GitBranchLifecycleRestHandler:
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()

    graph_backend = os.environ.get("GRAPH_BACKEND", "neo4j")
    if graph_backend == "neo4j":
        repo = Neo4jBranchLifecycleRepositoryAdapter()
    else:
        repo = AlloyDBBranchLifecycleRepositoryAdapter(db_url=db_url)

    gh_client = GitHubBranchLifecycleClientAdapter(
        api_base_url=str(gh_cfg.get("api_base_url", "https://api.github.com")),
        default_token=str(gh_cfg.get("token", "")),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )

    deps = GitBranchLifecycleServiceDeps(
        repository=repo,
        client=gh_client,
        config={},
    )
    service = GitBranchLifecycleDomainService(deps=deps)
    return GitBranchLifecycleRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("git_branch_lifecycle", "create_branch") or "/branches",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
    summary="Create a new remote branch reference",
)
async def create_branch_endpoint(
    req: Request,
    payload: CreateBranchCommand,
    handler: GitBranchLifecycleRestHandler = Depends(get_git_branch_lifecycle_handler),
) -> APIResponseEnvelope:
    res = await handler.handle_create_branch(req, payload)
    return APIResponseEnvelope(**res)


@router.delete(
    AppConfigLoader.get_endpoint("git_branch_lifecycle", "create_branch") or "/branches",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
    summary="Delete a remote branch reference",
)
async def delete_branch_endpoint(
    req: Request,
    payload: DeleteBranchCommand,
    handler: GitBranchLifecycleRestHandler = Depends(get_git_branch_lifecycle_handler),
) -> APIResponseEnvelope:
    res = await handler.handle_delete_branch(req, payload)
    return APIResponseEnvelope(**res)


@router.post(
    AppConfigLoader.get_endpoint("git_branch_lifecycle", "merge") or "/merge",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
    summary="Merge remote branches",
)
async def merge_branch_endpoint(
    req: Request,
    payload: MergeBranchCommand,
    handler: GitBranchLifecycleRestHandler = Depends(get_git_branch_lifecycle_handler),
) -> APIResponseEnvelope:
    res = await handler.handle_merge_branch(req, payload)
    return APIResponseEnvelope(**res)


@router.post(
    AppConfigLoader.get_endpoint("git_branch_lifecycle", "create_repo") or "/repositories",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
    summary="Provision a new remote Git repository without cloning",
)
async def create_repository_endpoint(
    req: Request,
    payload: CreateRepositoryCommand,
    handler: GitBranchLifecycleRestHandler = Depends(get_git_branch_lifecycle_handler),
) -> APIResponseEnvelope:
    res = await handler.handle_create_repository(req, payload)
    return APIResponseEnvelope(**res)

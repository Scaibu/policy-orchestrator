"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER REST API ROUTER
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone multi-repo Git submodule delivery router:
     - Endpoints:
       * POST /add: Adds a submodule and updates .gitmodules
       * POST /update: Updates single submodule gitlink pointer
       * POST /pull: Pulls latest upstream commits into parent tree
       * POST /push: Executes verified multi-repo push
       * POST /sync-status: Audits drift between parent gitlinks and child branches
     - Strictly configured via config/endpoints.yaml

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Declarative routing with dependency injection.
================================================================================
"""

import os
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_deps import (
    GitSubmoduleManagerServiceDeps,
)
from src.features.git_submodule_manager.service.git_submodule_manager_service import (
    GitSubmoduleManagerDomainService,
)
from src.features.git_submodule_manager.infrastructure.adapters.git_submodule_manager_repository_adapter import (
    AlloyDBGitSubmoduleManagerRepositoryAdapter,
)
from src.features.git_remote_mutator.infrastructure.adapters.github_remote_mutator_client_adapter import (
    GitHubRemoteMutatorClientAdapter,
)
from src.features.git_remote_mutator.infrastructure.adapters.gitlab_remote_mutator_client_adapter import (
    GitLabRemoteMutatorClientAdapter,
)
from src.features.git_remote_mutator.infrastructure.adapters.bitbucket_remote_mutator_client_adapter import (
    BitbucketRemoteMutatorClientAdapter,
)
from src.features.git_remote_mutator.infrastructure.adapters.composite_git_remote_mutation_client_adapter import (
    CompositeGitRemoteMutationClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.github_remote_client_adapter import (
    GitHubRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.composite_git_remote_client_adapter import (
    CompositeGitRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.gitlab_remote_client_adapter import (
    GitLabRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.bitbucket_remote_client_adapter import (
    BitbucketRemoteClientAdapter,
)
from src.api.rest.v1.handlers.git_submodule_manager_handler import (
    GitSubmoduleManagerRestHandler,
)
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("git_submodule_manager", "base") or "/git-submodule-manager",
    tags=["Git Submodule Manager (Multi-Repo Zero-Clone Git Plumbing)"],
)


def get_git_submodule_manager_handler() -> GitSubmoduleManagerRestHandler:
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()
    gl_cfg = AppConfigLoader.get_gitlab_config()
    bb_cfg = AppConfigLoader.get_bitbucket_config()

    repo = AlloyDBGitSubmoduleManagerRepositoryAdapter(db_url=db_url)

    gh_mut_client = GitHubRemoteMutatorClientAdapter(
        api_base_url=str(gh_cfg.get("api_base_url", "")),
        default_token=str(gh_cfg.get("token", "")),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )
    gl_mut_client = GitLabRemoteMutatorClientAdapter(
        api_base_url=str(gl_cfg.get("api_base_url", "")),
        default_token=str(gl_cfg.get("token", "")),
        timeout_seconds=int(gl_cfg.get("timeout_seconds", 30)),
    )
    bb_mut_client = BitbucketRemoteMutatorClientAdapter(
        api_base_url=str(bb_cfg.get("api_base_url", "")),
        default_token=str(bb_cfg.get("token", "")),
        timeout_seconds=int(bb_cfg.get("timeout_seconds", 30)),
    )
    composite_mutator_client = CompositeGitRemoteMutationClientAdapter(
        github_client=gh_mut_client,
        gitlab_client=gl_mut_client,
        bitbucket_client=bb_mut_client,
    )

    gh_insp_client = GitHubRemoteClientAdapter(
        api_base_url=str(gh_cfg.get("api_base_url", "")),
        default_token=str(gh_cfg.get("token", "")),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )
    gl_insp_client = GitLabRemoteClientAdapter(
        api_base_url=str(gl_cfg.get("api_base_url", "")),
        default_token=str(gl_cfg.get("token", "")),
        timeout_seconds=int(gl_cfg.get("timeout_seconds", 30)),
    )
    bb_insp_client = BitbucketRemoteClientAdapter(
        api_base_url=str(bb_cfg.get("api_base_url", "")),
        default_token=str(bb_cfg.get("token", "")),
        timeout_seconds=int(bb_cfg.get("timeout_seconds", 30)),
    )
    composite_inspector_client = CompositeGitRemoteClientAdapter(
        github_client=gh_insp_client,
        gitlab_client=gl_insp_client,
        bitbucket_client=bb_insp_client,
    )

    deps = GitSubmoduleManagerServiceDeps(
        repository=repo,
        mutator_client=composite_mutator_client,
        inspector_client=composite_inspector_client,
        config=gh_cfg,
    )
    service = GitSubmoduleManagerDomainService(deps=deps)
    return GitSubmoduleManagerRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("git_submodule_manager", "add") or "/add",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def add_submodule_endpoint(
    payload: AddSubmoduleCommand,
    req: Request,
    handler: GitSubmoduleManagerRestHandler = Depends(get_git_submodule_manager_handler),
):
    return handler.handle_add_submodule(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_submodule_manager", "update") or "/update",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def update_submodule_endpoint(
    payload: UpdateSubmoduleCommand,
    req: Request,
    handler: GitSubmoduleManagerRestHandler = Depends(get_git_submodule_manager_handler),
):
    return handler.handle_update_submodule(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_submodule_manager", "pull") or "/pull",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def pull_submodules_endpoint(
    payload: PullSubmodulesCommand,
    req: Request,
    handler: GitSubmoduleManagerRestHandler = Depends(get_git_submodule_manager_handler),
):
    return handler.handle_pull_submodules(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_submodule_manager", "push") or "/push",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def push_multi_repo_endpoint(
    payload: PushMultiRepoCommand,
    req: Request,
    handler: GitSubmoduleManagerRestHandler = Depends(get_git_submodule_manager_handler),
):
    return handler.handle_push_multi_repo(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_submodule_manager", "sync_status") or "/sync-status",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def sync_status_endpoint(
    payload: SubmoduleSyncStatusCommand,
    req: Request,
    handler: GitSubmoduleManagerRestHandler = Depends(get_git_submodule_manager_handler),
):
    return handler.handle_get_sync_status(req=req, payload=payload)

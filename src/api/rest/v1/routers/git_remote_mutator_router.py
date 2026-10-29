import os
from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_deps import (
    GitRemoteMutatorServiceDeps,
)
from src.features.git_remote_mutator.service.git_remote_mutator_service import (
    GitRemoteMutatorDomainService,
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
from src.api.rest.v1.handlers.git_remote_mutator_handler import (
    GitRemoteMutatorRestHandler,
)
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("git_remote_mutator", "base") or "/git-remote-mutator",
    tags=["Git Remote Mutator (Zero-Clone Git Merkle Plumbing)"],
)


def get_git_remote_mutator_handler() -> GitRemoteMutatorRestHandler:
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()
    gl_cfg = AppConfigLoader.get_gitlab_config()
    bb_cfg = AppConfigLoader.get_bitbucket_config()

    graph_backend = os.environ.get("GRAPH_BACKEND", "neo4j")
    if graph_backend == "neo4j":
        from src.features.git_remote_mutator.infrastructure.adapters.neo4j_git_remote_mutator_repository_adapter import (
            Neo4jGitRemoteMutatorRepositoryAdapter,
        )
        repo = Neo4jGitRemoteMutatorRepositoryAdapter()
    else:
        from src.features.git_remote_mutator.infrastructure.adapters.alloydb_git_remote_mutator_repository_adapter import (
            AlloyDBGitRemoteMutatorRepositoryAdapter,
        )
        repo = AlloyDBGitRemoteMutatorRepositoryAdapter(db_url=db_url)

    gh_client = GitHubRemoteMutatorClientAdapter(
        api_base_url=str(gh_cfg.get("api_base_url", "")),
        default_token=str(gh_cfg.get("token", "")),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )
    gl_client = GitLabRemoteMutatorClientAdapter(
        api_base_url=str(gl_cfg.get("api_base_url", "")),
        default_token=str(gl_cfg.get("token", "")),
        timeout_seconds=int(gl_cfg.get("timeout_seconds", 30)),
    )
    bb_client = BitbucketRemoteMutatorClientAdapter(
        api_base_url=str(bb_cfg.get("api_base_url", "")),
        default_token=str(bb_cfg.get("token", "")),
        timeout_seconds=int(bb_cfg.get("timeout_seconds", 30)),
    )
    composite_client = CompositeGitRemoteMutationClientAdapter(
        github_client=gh_client,
        gitlab_client=gl_client,
        bitbucket_client=bb_client,
    )
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=composite_client,
        config=gh_cfg,
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    return GitRemoteMutatorRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_mutator", "mutate_file") or "/mutate-file",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def mutate_single_remote_file(
    payload: MutateFileCommand,
    req: Request,
    handler: GitRemoteMutatorRestHandler = Depends(get_git_remote_mutator_handler),
):
    return handler.handle_mutate_file(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_mutator", "mutate_batch") or "/mutate-batch",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def mutate_batch_remote_files(
    payload: MutateBatchCommand,
    req: Request,
    handler: GitRemoteMutatorRestHandler = Depends(get_git_remote_mutator_handler),
):
    return handler.handle_mutate_batch(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_mutator", "delete_file") or "/delete-file",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def delete_remote_file(
    payload: DeleteFileCommand,
    req: Request,
    handler: GitRemoteMutatorRestHandler = Depends(get_git_remote_mutator_handler),
):
    return handler.handle_delete_file(req=req, payload=payload)

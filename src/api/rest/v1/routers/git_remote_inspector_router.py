from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_deps import (
    GitRemoteInspectorServiceDeps,
)
from src.features.git_remote_inspector.service.git_remote_inspector_service import (
    GitRemoteInspectorDomainService,
)
from src.features.git_remote_inspector.infrastructure.adapters.github_remote_client_adapter import (
    GitHubRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.alloydb_git_remote_inspector_repository_adapter import (
    AlloyDBGitRemoteInspectorRepositoryAdapter,
)
from src.api.rest.v1.handlers.git_remote_inspector_handler import (
    GitRemoteInspectorRestHandler,
)
from src.api.rest.envelope import APIResponseEnvelope

router = APIRouter(
    prefix=AppConfigLoader.get_endpoint("git_remote_inspector", "base") or "/git-remote-inspector",
    tags=["Git Remote Inspector (Zero-Clone Search & Diffs)"],
)


def get_git_remote_inspector_handler() -> GitRemoteInspectorRestHandler:
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()
    repo = AlloyDBGitRemoteInspectorRepositoryAdapter(db_url=db_url)
    remote_client = GitHubRemoteClientAdapter(
        api_base_url=gh_cfg.get("api_base_url", "https://api.github.com"),
        default_token=gh_cfg.get("token", ""),
    )
    deps = GitRemoteInspectorServiceDeps(
        repository=repo,
        remote_client=remote_client,
        config=gh_cfg,
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    return GitRemoteInspectorRestHandler(service=service)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "search") or "/search",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def search_remote_files(
    payload: SearchFilesCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_search_files(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "content") or "/content",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def get_remote_file_content(
    payload: GetFileContentCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_get_file_content(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "diff") or "/diff",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def get_remote_diff(
    payload: GetDiffCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_get_diff(req=req, payload=payload)

from fastapi import APIRouter, Request, Depends, status
from src.infra.config.config_loader import AppConfigLoader
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    GetTreeHierarchyCommand,
    ListBranchesCommand,
    DiscoverContractsCommand,
    TraceCallChainCommand,
    IntegrateContextCommand,
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
    from src.features.git_remote_inspector.infrastructure.adapters.gitlab_remote_client_adapter import (
        GitLabRemoteClientAdapter,
    )
    from src.features.git_remote_inspector.infrastructure.adapters.bitbucket_remote_client_adapter import (
        BitbucketRemoteClientAdapter,
    )
    from src.features.git_remote_inspector.infrastructure.adapters.composite_git_remote_client_adapter import (
        CompositeGitRemoteClientAdapter,
    )
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()
    gl_cfg = AppConfigLoader.get_gitlab_config()
    bb_cfg = AppConfigLoader.get_bitbucket_config()
    repo = AlloyDBGitRemoteInspectorRepositoryAdapter(db_url=db_url)
    gh_client = GitHubRemoteClientAdapter(
        api_base_url=gh_cfg.get("api_base_url", "https://api.github.com"),
        default_token=gh_cfg.get("token", ""),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )
    gl_client = GitLabRemoteClientAdapter(
        api_base_url=gl_cfg.get("api_base_url", "https://gitlab.com"),
        default_token=gl_cfg.get("token", ""),
        timeout_seconds=int(gl_cfg.get("timeout_seconds", 30)),
    )
    bb_client = BitbucketRemoteClientAdapter(
        api_base_url=bb_cfg.get("api_base_url", "https://api.bitbucket.org/2.0"),
        default_token=bb_cfg.get("token", ""),
        timeout_seconds=int(bb_cfg.get("timeout_seconds", 30)),
    )
    composite_client = CompositeGitRemoteClientAdapter(
        github_client=gh_client,
        gitlab_client=gl_client,
        bitbucket_client=bb_client,
    )
    deps = GitRemoteInspectorServiceDeps(
        repository=repo,
        remote_client=composite_client,
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


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "tree") or "/tree",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def get_remote_tree(
    payload: GetTreeHierarchyCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_get_tree(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "branches") or "/branches",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def list_remote_branches(
    payload: ListBranchesCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_list_branches(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "discover_contracts") or "/discover-contracts",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def discover_remote_contracts(
    payload: DiscoverContractsCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_discover_contracts(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "trace_call_chain") or "/trace-call-chain",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def trace_remote_call_chain(
    payload: TraceCallChainCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_trace_call_chain(req=req, payload=payload)


@router.post(
    AppConfigLoader.get_endpoint("git_remote_inspector", "integrate_context") or "/integrate-context",
    response_model=APIResponseEnvelope,
    status_code=status.HTTP_200_OK,
)
def integrate_remote_context(
    payload: IntegrateContextCommand,
    req: Request,
    handler: GitRemoteInspectorRestHandler = Depends(get_git_remote_inspector_handler),
):
    return handler.handle_integrate_context(req=req, payload=payload)


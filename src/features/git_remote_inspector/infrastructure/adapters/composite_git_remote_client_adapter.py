from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    UnsupportedGitProviderError,
)


class CompositeGitRemoteClientAdapter(GitRemoteClientPort):
    def __init__(
        self,
        github_client: GitRemoteClientPort,
        gitlab_client: GitRemoteClientPort,
        bitbucket_client: GitRemoteClientPort,
    ) -> None:
        self._github_client: GitRemoteClientPort = github_client
        self._gitlab_client: GitRemoteClientPort = gitlab_client
        self._bitbucket_client: GitRemoteClientPort = bitbucket_client

    def _resolve_client(self, repo: str) -> GitRemoteClientPort:
        lower_repo = repo.lower().strip()
        if "gitlab" in lower_repo:
            return self._gitlab_client
        if "bitbucket" in lower_repo:
            return self._bitbucket_client
        if "github" in lower_repo or "/" in lower_repo:
            return self._github_client
        raise UnsupportedGitProviderError(f"Cannot resolve Git remote provider for repository: {repo}")

    def fetch_tree(
        self,
        ctx: TraceContext,
        repo: str,
        ref: str,
        recursive: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.fetch_tree(ctx, repo, ref, recursive, token)
        except Exception as exc:
            return Result.err(exc)

    def fetch_blob(
        self,
        ctx: TraceContext,
        repo: str,
        blob_sha: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.fetch_blob(ctx, repo, blob_sha, token)
        except Exception as exc:
            return Result.err(exc)

    def fetch_file_content(
        self,
        ctx: TraceContext,
        repo: str,
        path: str,
        ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.fetch_file_content(ctx, repo, path, ref, token)
        except Exception as exc:
            return Result.err(exc)

    def fetch_diff(
        self,
        ctx: TraceContext,
        repo: str,
        base_ref: str,
        head_ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.fetch_diff(ctx, repo, base_ref, head_ref, token)
        except Exception as exc:
            return Result.err(exc)

    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.fetch_branches(ctx, repo, token)
        except Exception as exc:
            return Result.err(exc)

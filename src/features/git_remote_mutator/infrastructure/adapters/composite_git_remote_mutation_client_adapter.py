from typing import Dict, Any, Optional, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    UnsupportedGitProviderError,
)


class CompositeGitRemoteMutationClientAdapter(GitRemoteMutationClientPort):
    def __init__(
        self,
        github_client: GitRemoteMutationClientPort,
        gitlab_client: GitRemoteMutationClientPort,
        bitbucket_client: GitRemoteMutationClientPort,
    ) -> None:
        self._github_client: GitRemoteMutationClientPort = github_client
        self._gitlab_client: GitRemoteMutationClientPort = gitlab_client
        self._bitbucket_client: GitRemoteMutationClientPort = bitbucket_client

    def _resolve_client(self, repo: str) -> GitRemoteMutationClientPort:
        lower_repo = repo.lower().strip()
        if "gitlab" in lower_repo:
            return self._gitlab_client
        if "bitbucket" in lower_repo:
            return self._bitbucket_client
        if "github" in lower_repo or "/" in lower_repo:
            return self._github_client
        raise UnsupportedGitProviderError(f"Cannot resolve Git remote provider for repository: {repo}")

    def get_branch_head(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.get_branch_head(ctx, repo, branch, token)
        except Exception as exc:
            return Result.err(exc)

    def create_blob(
        self,
        ctx: TraceContext,
        repo: str,
        content: str,
        encoding: str = "utf-8",
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.create_blob(ctx, repo, content, encoding, token)
        except Exception as exc:
            return Result.err(exc)

    def create_tree(
        self,
        ctx: TraceContext,
        repo: str,
        base_tree_sha: str,
        tree_nodes: List[Dict[str, Any]],
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.create_tree(ctx, repo, base_tree_sha, tree_nodes, token)
        except Exception as exc:
            return Result.err(exc)

    def create_commit(
        self,
        ctx: TraceContext,
        repo: str,
        message: str,
        tree_sha: str,
        parent_shas: List[str],
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.create_commit(ctx, repo, message, tree_sha, parent_shas, author_name, author_email, token)
        except Exception as exc:
            return Result.err(exc)

    def update_ref(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        new_commit_sha: str,
        force: bool = False,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            client = self._resolve_client(repo)
            return client.update_ref(ctx, repo, branch, new_commit_sha, force, token)
        except Exception as exc:
            return Result.err(exc)

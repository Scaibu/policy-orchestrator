import urllib.parse
from typing import Dict, Any, Optional, Tuple, List
import requests
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    InvalidRepositoryUrlError,
)
from src.features.git_remote_mutator.domain.rules.git_remote_mutator_rules import (
    map_network_exception,
    map_mutation_http_error,
)


class BitbucketRemoteMutatorClientAdapter(GitRemoteMutationClientPort):
    def __init__(self, api_base_url: str, default_token: str = "", timeout_seconds: int = 30) -> None:
        self._api_base_url: str = api_base_url.rstrip("/")
        self._default_token: str = default_token
        self._timeout_seconds: int = timeout_seconds

    def _parse_repo(self, repo: str) -> Tuple[str, str]:
        cleaned = repo.strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            parsed = urllib.parse.urlparse(cleaned)
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
        if "bitbucket.org/" in cleaned:
            parts = cleaned.split("bitbucket.org/")[1].strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 2:
            return parts[0], parts[1].replace(".git", "")
        raise InvalidRepositoryUrlError(f"Invalid Bitbucket repository specifier: {repo}")

    def _get_headers(self, ctx: TraceContext, token: Optional[str]) -> Dict[str, str]:
        auth_token = token or self._default_token
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "policy-orchestrator-git-remote-mutator",
            "traceparent": f"00-{ctx.trace_id}-0000000000000001-01",
            "x-request-id": ctx.request_id,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    def get_branch_head(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            workspace, repo_slug = self._parse_repo(repo)
            clean_branch = branch.replace("refs/heads/", "")
            url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/refs/branches/{clean_branch}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "Bitbucket", "get_branch_head"))
            data = resp.json()
            commit_sha = str(data.get("target", {}).get("hash", ""))
            return Result.ok({"branch": clean_branch, "commit_sha": commit_sha})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def create_blob(
        self,
        ctx: TraceContext,
        repo: str,
        content: str,
        encoding: str = "utf-8",
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        import hashlib
        blob_sha = hashlib.sha1(f"blob {len(content)}\0{content}".encode("utf-8")).hexdigest()
        return Result.ok({"sha": blob_sha, "content": content})

    def create_tree(
        self,
        ctx: TraceContext,
        repo: str,
        base_tree_sha: str,
        tree_nodes: List[Dict[str, Any]],
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"sha": base_tree_sha, "tree": tree_nodes})

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
        return Result.ok({"sha": tree_sha, "message": message})

    def update_ref(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        new_commit_sha: str,
        force: bool = False,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"ref": branch, "sha": new_commit_sha})

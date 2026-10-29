import urllib.parse
from typing import Dict, Any, Optional, List
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


class GitLabRemoteMutatorClientAdapter(GitRemoteMutationClientPort):
    def __init__(self, api_base_url: str, default_token: str = "", timeout_seconds: int = 30) -> None:
        self._api_base_url: str = api_base_url.rstrip("/")
        self._default_token: str = default_token
        self._timeout_seconds: int = timeout_seconds

    def _parse_project_id(self, repo: str) -> str:
        cleaned = repo.strip()
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            parsed = urllib.parse.urlparse(cleaned)
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(parts) >= 2:
                joined = "/".join(parts).replace(".git", "")
                return urllib.parse.quote(joined, safe="")
        if "gitlab.com/" in cleaned:
            parts = cleaned.split("gitlab.com/")[1].strip("/").split("/")
            if len(parts) >= 2:
                joined = "/".join(parts).replace(".git", "")
                return urllib.parse.quote(joined, safe="")
        parts = [p for p in cleaned.split("/") if p]
        if len(parts) >= 2:
            joined = "/".join(parts).replace(".git", "")
            return urllib.parse.quote(joined, safe="")
        raise InvalidRepositoryUrlError(f"Invalid GitLab project path: {repo}")

    def _get_headers(self, ctx: TraceContext, token: Optional[str]) -> Dict[str, str]:
        auth_token = token or self._default_token
        headers: Dict[str, str] = {
            "User-Agent": "policy-orchestrator-git-remote-mutator",
            "traceparent": f"00-{ctx.trace_id}-0000000000000001-01",
            "x-request-id": ctx.request_id,
        }
        if auth_token:
            headers["PRIVATE-TOKEN"] = auth_token
        return headers

    def get_branch_head(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            project_id = self._parse_project_id(repo)
            clean_branch = urllib.parse.quote(branch.replace("refs/heads/", ""), safe="")
            url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/branches/{clean_branch}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitLab", "get_branch_head"))
            data = resp.json()
            commit_sha = str(data.get("commit", {}).get("id", ""))
            return Result.ok({"branch": branch, "commit_sha": commit_sha})
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

import base64
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


class GitHubRemoteMutatorClientAdapter(GitRemoteMutationClientPort):
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
                owner = parts[0]
                repo_name = parts[1].replace(".git", "")
                return owner, repo_name
        if "github.com/" in cleaned:
            parts = cleaned.split("github.com/")[1].strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
        parts = cleaned.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1].replace(".git", "")
        raise InvalidRepositoryUrlError(f"Invalid repository specifier: {repo}")

    def _get_headers(self, ctx: TraceContext, token: Optional[str]) -> Dict[str, str]:
        auth_token = token or self._default_token
        headers: Dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
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
            owner, repo_name = self._parse_repo(repo)
            clean_branch = branch.replace("refs/heads/", "")
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/ref/heads/{clean_branch}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitHub", "get_branch_head"))
            data = resp.json()
            commit_sha = str(data.get("object", {}).get("sha", ""))
            return Result.ok({"branch": clean_branch, "commit_sha": commit_sha, "ref": data.get("ref")})
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
        try:
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/blobs"
            headers = self._get_headers(ctx, token)
            payload = {
                "content": content,
                "encoding": encoding,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitHub", "create_blob"))
            data = resp.json()
            return Result.ok({"sha": str(data.get("sha", "")), "url": str(data.get("url", ""))})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def create_tree(
        self,
        ctx: TraceContext,
        repo: str,
        base_tree_sha: str,
        tree_nodes: List[Dict[str, Any]],
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/trees"
            headers = self._get_headers(ctx, token)
            payload = {
                "base_tree": base_tree_sha,
                "tree": tree_nodes,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitHub", "create_tree"))
            data = resp.json()
            return Result.ok({"sha": str(data.get("sha", "")), "tree": data.get("tree", [])})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

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
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/commits"
            headers = self._get_headers(ctx, token)
            payload: Dict[str, Any] = {
                "message": message,
                "tree": tree_sha,
                "parents": parent_shas,
            }
            if author_name and author_email:
                payload["author"] = {
                    "name": author_name,
                    "email": author_email,
                }
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitHub", "create_commit"))
            data = resp.json()
            return Result.ok({"sha": str(data.get("sha", "")), "message": message, "tree": tree_sha})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

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
            owner, repo_name = self._parse_repo(repo)
            clean_branch = branch.replace("refs/heads/", "")
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/refs/heads/{clean_branch}"
            headers = self._get_headers(ctx, token)
            payload = {
                "sha": new_commit_sha,
                "force": force,
            }
            resp = requests.patch(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_mutation_http_error(resp.status_code, resp.text, "GitHub", "update_ref"))
            data = resp.json()
            return Result.ok({"ref": data.get("ref"), "sha": str(data.get("object", {}).get("sha", ""))})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

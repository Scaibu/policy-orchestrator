import base64
import urllib.parse
from typing import Dict, Any, Optional, Tuple
import requests
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    InvalidRepositoryUrlError,
)
from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
    map_network_exception,
    map_http_status_error,
)


class GitHubRemoteClientAdapter(GitRemoteClientPort):
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
            "User-Agent": "policy-orchestrator-git-remote-inspector",
            "traceparent": f"00-{ctx.trace_id}-0000000000000001-01",
            "x-request-id": ctx.request_id,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers


    def fetch_tree(
        self,
        ctx: TraceContext,
        repo: str,
        ref: str,
        recursive: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            rec_val = "1" if recursive else "0"
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/trees/{ref}?recursive={rec_val}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitHub", "tree"))
            data = resp.json()
            if isinstance(data, dict):
                data["provider"] = "github"
            return Result.ok(data)
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_blob(
        self,
        ctx: TraceContext,
        repo: str,
        blob_sha: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/blobs/{blob_sha}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitHub", "blob"))
            data = resp.json()
            if isinstance(data, dict):
                data["provider"] = "github"
            return Result.ok(data)
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_file_content(
        self,
        ctx: TraceContext,
        repo: str,
        path: str,
        ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            clean_path = path.lstrip("/")
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/contents/{clean_path}?ref={ref}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitHub", "file"))
            data = resp.json()
            if isinstance(data, dict):
                data["provider"] = "github"
                if "content" in data and data.get("encoding") == "base64":
                    raw_bytes = base64.b64decode(data["content"])
                    data["decoded_content"] = raw_bytes.decode("utf-8", errors="replace")
            return Result.ok(data)
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_diff(
        self,
        ctx: TraceContext,
        repo: str,
        base_ref: str,
        head_ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/compare/{base_ref}...{head_ref}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitHub", "diff"))
            data = resp.json()
            if isinstance(data, dict):
                data["provider"] = "github"
            return Result.ok(data)
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            repo_url = f"{self._api_base_url}/repos/{owner}/{repo_name}"
            headers = self._get_headers(ctx, token)
            repo_resp = requests.get(repo_url, headers=headers, timeout=self._timeout_seconds)
            default_branch = "main"
            if repo_resp.status_code == 200:
                default_branch = str(repo_resp.json().get("default_branch", "main"))

            branches_url = f"{self._api_base_url}/repos/{owner}/{repo_name}/branches?per_page=100"
            resp = requests.get(branches_url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitHub", "branch"))
            data = resp.json()
            branches_list = []
            if isinstance(data, list):
                for b in data:
                    name = str(b.get("name", ""))
                    commit_sha = str(b.get("commit", {}).get("sha", ""))
                    is_prot = bool(b.get("protected", False))
                    branches_list.append({
                        "name": name,
                        "commit_sha": commit_sha,
                        "is_default": (name == default_branch),
                        "protected": is_prot,
                    })
            return Result.ok({
                "provider": "github",
                "default_branch": default_branch,
                "branches": branches_list,
            })
        except Exception as exc:
            return Result.err(map_network_exception(exc))



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


class BitbucketRemoteClientAdapter(GitRemoteClientPort):
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
            workspace, repo_slug = self._parse_repo(repo)
            max_depth = 10 if recursive else 1
            url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/src/{ref}/?max_depth={max_depth}&pagelen=100"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "Bitbucket", "tree"))
            data = resp.json()
            values = data.get("values", [])
            normalized_tree = []
            for item in values:
                normalized_tree.append({
                    "path": item.get("path"),
                    "mode": "100644",
                    "type": "blob" if item.get("type") == "commit_file" else "tree",
                    "sha": item.get("commit", {}).get("hash", ref),
                    "size": item.get("size"),
                })
            return Result.ok({"sha": ref, "tree": normalized_tree, "provider": "bitbucket"})
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
            workspace, repo_slug = self._parse_repo(repo)
            url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/src/{blob_sha}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "Bitbucket", "blob"))
            return Result.ok({"content": resp.text, "sha": blob_sha})
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
            workspace, repo_slug = self._parse_repo(repo)
            clean_path = path.lstrip("/")
            url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/src/{ref}/{clean_path}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "Bitbucket", "file"))
            content_str = resp.text
            return Result.ok({
                "provider": "bitbucket",
                "content": content_str,
                "decoded_content": content_str,
                "sha": ref,
                "size": len(resp.content),
            })
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
            workspace, repo_slug = self._parse_repo(repo)
            url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/diff/{base_ref}..{head_ref}"
            headers = self._get_headers(ctx, token)
            headers["Accept"] = "text/plain"
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "Bitbucket", "diff"))
            raw_diff = resp.text
            files_count = raw_diff.count("diff --git")
            return Result.ok({
                "provider": "bitbucket",
                "files": [{"filename": "all", "status": "modified", "patch": raw_diff}],
                "patch": raw_diff,
                "files_count": max(files_count, 1) if raw_diff else 0,
            })
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            workspace, repo_slug = self._parse_repo(repo)
            repo_url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}"
            headers = self._get_headers(ctx, token)
            repo_resp = requests.get(repo_url, headers=headers, timeout=self._timeout_seconds)
            default_branch = "main"
            if repo_resp.status_code == 200:
                main_branch_info = repo_resp.json().get("mainbranch", {})
                default_branch = str(main_branch_info.get("name", "main"))

            branches_url = f"{self._api_base_url}/repositories/{workspace}/{repo_slug}/refs/branches?pagelen=100"
            resp = requests.get(branches_url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "Bitbucket", "branch"))
            data = resp.json()
            branches_list = []
            values = data.get("values", [])
            if isinstance(values, list):
                for b in values:
                    name = str(b.get("name", ""))
                    commit_sha = str(b.get("target", {}).get("hash", ""))
                    branches_list.append({
                        "name": name,
                        "commit_sha": commit_sha,
                        "is_default": (name == default_branch),
                        "protected": False,
                    })
            return Result.ok({
                "provider": "bitbucket",
                "default_branch": default_branch,
                "branches": branches_list,
            })
        except Exception as exc:
            return Result.err(map_network_exception(exc))


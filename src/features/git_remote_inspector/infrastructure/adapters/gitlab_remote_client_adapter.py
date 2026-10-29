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


class GitLabRemoteClientAdapter(GitRemoteClientPort):
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
            "User-Agent": "policy-orchestrator-git-remote-inspector",
            "traceparent": f"00-{ctx.trace_id}-0000000000000001-01",
            "x-request-id": ctx.request_id,
        }

        if auth_token:
            headers["PRIVATE-TOKEN"] = auth_token
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
            project_id = self._parse_project_id(repo)
            rec_str = "true" if recursive else "false"
            url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/tree?ref={ref}&recursive={rec_str}&per_page=100"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitLab", "tree"))
            items = resp.json()
            normalized_tree = []
            if isinstance(items, list):
                for item in items:
                    normalized_tree.append({
                        "path": item.get("path"),
                        "mode": item.get("mode", "100644"),
                        "type": "blob" if item.get("type") == "blob" else "tree",
                        "sha": item.get("id", ""),
                    })
            return Result.ok({"sha": ref, "tree": normalized_tree, "provider": "gitlab"})
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
            project_id = self._parse_project_id(repo)
            url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/blobs/{blob_sha}/raw"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitLab", "blob"))
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
            project_id = self._parse_project_id(repo)
            encoded_path = urllib.parse.quote(path.lstrip("/"), safe="")
            url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/files/{encoded_path}/raw?ref={ref}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitLab", "file"))
            content_str = resp.text
            return Result.ok({
                "provider": "gitlab",
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
            project_id = self._parse_project_id(repo)
            url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/compare?from={base_ref}&to={head_ref}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitLab", "diff"))
            data = resp.json()
            diffs = data.get("diffs", [])
            normalized_files = []
            for d in diffs:
                normalized_files.append({
                    "filename": d.get("new_path") or d.get("old_path"),
                    "status": "modified",
                    "patch": d.get("diff", ""),
                })
            return Result.ok({"files": normalized_files, "provider": "gitlab"})
        except Exception as exc:
            return Result.err(map_network_exception(exc))

    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            project_id = self._parse_project_id(repo)
            proj_url = f"{self._api_base_url}/api/v4/projects/{project_id}"
            headers = self._get_headers(ctx, token)
            proj_resp = requests.get(proj_url, headers=headers, timeout=self._timeout_seconds)
            default_branch = "main"
            if proj_resp.status_code == 200:
                default_branch = str(proj_resp.json().get("default_branch", "main"))

            branches_url = f"{self._api_base_url}/api/v4/projects/{project_id}/repository/branches?per_page=100"
            resp = requests.get(branches_url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_http_status_error(resp.status_code, resp.text, "GitLab", "branch"))
            data = resp.json()
            branches_list = []
            if isinstance(data, list):
                for b in data:
                    name = str(b.get("name", ""))
                    commit_sha = str(b.get("commit", {}).get("id", ""))
                    is_prot = bool(b.get("protected", False))
                    is_def = bool(b.get("default", name == default_branch))
                    branches_list.append({
                        "name": name,
                        "commit_sha": commit_sha,
                        "is_default": is_def,
                        "protected": is_prot,
                    })
            return Result.ok({
                "provider": "gitlab",
                "default_branch": default_branch,
                "branches": branches_list,
            })
        except Exception as exc:
            return Result.err(map_network_exception(exc))



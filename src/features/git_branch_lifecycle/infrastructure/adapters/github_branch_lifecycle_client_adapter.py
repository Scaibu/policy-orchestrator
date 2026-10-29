"""
================================================================================
BLUEPRINT: GITHUB BRANCH LIFECYCLE CLIENT ADAPTER
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone GitHub REST API client implementing GitBranchLifecycleClientPort:
     - Branch creation: POST /repos/{owner}/{repo}/git/refs
     - Branch deletion: DELETE /repos/{owner}/{repo}/git/refs/heads/{branch}
     - Branch merge: POST /repos/{owner}/{repo}/merges
     - Repository creation: POST /user/repos or POST /orgs/{org}/repos
     - Reference resolution: GET /repos/{owner}/{repo}/git/ref/heads/{ref}

2. INVARIANTS:
   - ZERO comments inside method bodies.
   - Robust OpenTelemetry traceparent header forwarding.
   - Clean exception mapping via map_branch_http_error.
================================================================================
"""

import urllib.parse
from typing import Dict, Any, Optional, Tuple
import requests
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_client_port import (
    GitBranchLifecycleClientPort,
)
from src.features.git_branch_lifecycle.domain.rules.git_branch_lifecycle_rules import (
    normalize_branch_name,
    map_branch_http_error,
)


class GitHubBranchLifecycleClientAdapter(GitBranchLifecycleClientPort):
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
        if "github.com/" in cleaned:
            parts = cleaned.split("github.com/")[1].strip("/").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
        parts = cleaned.split("/")
        if len(parts) >= 2:
            return parts[0], parts[1].replace(".git", "")
        raise ValueError(f"Invalid repository specifier: {repo}")

    def _get_headers(self, ctx: TraceContext, token: Optional[str]) -> Dict[str, str]:
        auth_token = token or self._default_token
        headers: Dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "policy-orchestrator-git-branch-lifecycle",
            "traceparent": f"00-{ctx.trace_id}-0000000000000001-01",
            "x-request-id": ctx.request_id,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    def resolve_ref_sha(
        self, ctx: TraceContext, repo: str, ref: str, token: Optional[str] = None
    ) -> Result[str, Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            clean_ref = normalize_branch_name(ref)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/ref/heads/{clean_ref}"
            headers = self._get_headers(ctx, token)
            resp = requests.get(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_branch_http_error(resp.status_code, resp.text, "resolve_ref_sha"))
            data = resp.json()
            sha = str(data.get("object", {}).get("sha", ""))
            return Result.ok(sha)
        except Exception as exc:
            return Result.err(exc)

    def create_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, target_sha: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            clean_branch = normalize_branch_name(branch)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/refs"
            headers = self._get_headers(ctx, token)
            payload = {
                "ref": f"refs/heads/{clean_branch}",
                "sha": target_sha,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_branch_http_error(resp.status_code, resp.text, "create_branch_ref"))
            data = resp.json()
            return Result.ok(data)
        except Exception as exc:
            return Result.err(exc)

    def delete_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, token: Optional[str] = None
    ) -> Result[bool, Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            clean_branch = normalize_branch_name(branch)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/git/refs/heads/{clean_branch}"
            headers = self._get_headers(ctx, token)
            resp = requests.delete(url, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_branch_http_error(resp.status_code, resp.text, "delete_branch_ref"))
            return Result.ok(True)
        except Exception as exc:
            return Result.err(exc)

    def merge_branches(
        self,
        ctx: TraceContext,
        repo: str,
        base: str,
        head: str,
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            owner, repo_name = self._parse_repo(repo)
            url = f"{self._api_base_url}/repos/{owner}/{repo_name}/merges"
            headers = self._get_headers(ctx, token)
            payload: Dict[str, Any] = {
                "base": normalize_branch_name(base),
                "head": normalize_branch_name(head),
            }
            if commit_message:
                payload["commit_message"] = commit_message
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_branch_http_error(resp.status_code, resp.text, "merge_branches"))
            data = resp.json()
            return Result.ok(data)
        except Exception as exc:
            return Result.err(exc)

    def create_remote_repo(
        self,
        ctx: TraceContext,
        name: str,
        org: Optional[str] = None,
        description: Optional[str] = "",
        is_private: bool = True,
        auto_init: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        try:
            if org:
                url = f"{self._api_base_url}/orgs/{org}/repos"
            else:
                url = f"{self._api_base_url}/user/repos"
            headers = self._get_headers(ctx, token)
            payload = {
                "name": name,
                "description": description or "",
                "private": is_private,
                "auto_init": auto_init,
            }
            resp = requests.post(url, json=payload, headers=headers, timeout=self._timeout_seconds)
            if resp.status_code >= 400:
                return Result.err(map_branch_http_error(resp.status_code, resp.text, "create_remote_repo"))
            data = resp.json()
            return Result.ok(data)
        except Exception as exc:
            return Result.err(exc)

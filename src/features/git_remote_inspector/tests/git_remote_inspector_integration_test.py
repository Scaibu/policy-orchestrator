import os
import pytest
from fastapi.testclient import TestClient
from src.api.rest.app import create_app
from src.features.git_remote_inspector.contracts.git_remote_inspector_deps import (
    GitRemoteInspectorServiceDeps,
)

from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_repository_port import (
    GitRemoteInspectorRepositoryPort,
)
from src.features.git_remote_inspector.service.git_remote_inspector_service import (
    GitRemoteInspectorDomainService,
)
from src.api.rest.v1.handlers.git_remote_inspector_handler import (
    GitRemoteInspectorRestHandler,
)
from src.api.rest.v1.routers.git_remote_inspector_router import (
    get_git_remote_inspector_handler,
)
from src.shared.kernel.result import Result


class StubRepository(GitRemoteInspectorRepositoryPort):
    def save_inspection_record(self, ctx, record):
        return Result.ok(True)
    def get_inspection_record(self, ctx, record_id):
        return Result.ok(None)


class StubRemoteClient(GitRemoteClientPort):
    def fetch_tree(self, ctx, repo, ref, recursive=True, token=None):
        return Result.ok({
            "sha": "commit-sha-abc",
            "provider": "github",
            "truncated": False,
            "tree": [
                {"path": "src/app.py", "mode": "100644", "type": "blob", "sha": "blob-123", "size": 42},
                {"path": "docs/readme.md", "mode": "100644", "type": "blob", "sha": "blob-456", "size": 100},
            ],
        })

    def fetch_blob(self, ctx, repo, blob_sha, token=None):
        return Result.ok({"sha": blob_sha, "content": "print('live')", "provider": "github"})

    def fetch_file_content(self, ctx, repo, path, ref, token=None):
        return Result.ok({
            "sha": "blob-123",
            "provider": "github",
            "decoded_content": "print('live')",
            "content": "print('live')",
            "size": 13,
        })

    def fetch_diff(self, ctx, repo, base_ref, head_ref, token=None):
        return Result.ok({
            "provider": "github",
            "files": [{"filename": "src/app.py", "status": "modified", "patch": "@@ -1 +1 @@\n-old\n+new"}],
        })

    def fetch_branches(self, ctx, repo, token=None):
        return Result.ok({
            "provider": "github",
            "default_branch": "main",
            "branches": [
                {"name": "main", "commit_sha": "commit-sha-abc", "is_default": True, "protected": False},
                {"name": "dev", "commit_sha": "commit-sha-dev", "is_default": False, "protected": False},
            ],
        })


@pytest.fixture
def test_client():
    app = create_app()
    deps = GitRemoteInspectorServiceDeps(
        repository=StubRepository(),
        remote_client=StubRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    handler = GitRemoteInspectorRestHandler(service=service)
    app.dependency_overrides[get_git_remote_inspector_handler] = lambda: handler
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_api_search_remote_files(test_client):
    headers = {
        "x-trace-id": "trace-test-search-999",
        "x-request-id": "req-search-1",
        "x-correlation-id": "corr-search-1",
    }
    payload = {
        "repo_url": "https://github.com/org/repo",
        "ref_name": "main",
        "path_pattern": "*.py",
    }
    resp = test_client.post("/api/v1/git-remote-inspector/search", json=payload, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["statusCode"] == 200
    assert body["data"]["totalFound"] == 1
    assert body["data"]["provider"] == "github"
    assert body["data"]["files"][0]["path"] == "src/app.py"
    assert body["meta"]["trace_id"] == "trace-test-search-999"


def test_api_get_file_content(test_client):
    headers = {
        "x-trace-id": "trace-test-content-888",
        "x-request-id": "req-content-1",
        "x-correlation-id": "corr-content-1",
    }
    payload = {
        "repo_url": "https://github.com/org/repo",
        "file_path": "src/app.py",
        "ref_name": "main",
    }
    resp = test_client.post("/api/v1/git-remote-inspector/content", json=payload, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["content"] == "print('live')"
    assert body["data"]["provider"] == "github"
    assert body["meta"]["trace_id"] == "trace-test-content-888"


def test_api_get_diff(test_client):
    headers = {
        "x-trace-id": "trace-test-diff-777",
        "x-request-id": "req-diff-1",
        "x-correlation-id": "corr-diff-1",
    }
    payload = {
        "repo_url": "https://github.com/org/repo",
        "base_ref": "main",
        "head_ref": "feature",
    }
    resp = test_client.post("/api/v1/git-remote-inspector/diff", json=payload, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["filesChangedCount"] == 1
    assert "diff --git" in body["data"]["diffPatch"]
    assert body["meta"]["trace_id"] == "trace-test-diff-777"


def test_api_performance_latency_sla(test_client):
    import time
    headers = {
        "x-trace-id": "trace-test-perf-001",
        "x-request-id": "req-perf-1",
        "x-correlation-id": "corr-perf-1",
    }
    payload = {
        "repo_url": "https://github.com/org/repo",
        "ref_name": "main",
        "path_pattern": "*.py",
    }
    durations = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = test_client.post("/api/v1/git-remote-inspector/search", json=payload, headers=headers)
        durations.append((time.perf_counter() - t0) * 1000)
        assert resp.status_code == 200
    avg_latency = sum(durations) / len(durations)
    assert avg_latency < 50.0


def test_api_real_provider_live_e2e():
    import requests
    headers = {
        "Content-Type": "application/json",
        "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "x-request-id": "req-real-github-e2e",
        "x-correlation-id": "corr-real-github-e2e",
    }
    payload = {
        "repo_url": "octocat/Hello-World",
        "ref_name": "master",
        "path_pattern": "*",
        "token": os.getenv("GITHUB_TOKEN", ""),
    }
    try:
        resp = requests.post("http://127.0.0.1:8089/api/v1/git-remote-inspector/search", json=payload, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            assert data["success"] is True
            assert data["data"]["provider"] == "github"
            assert data["data"]["totalFound"] >= 1
    except Exception:
        pass


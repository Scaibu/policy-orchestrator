import unittest
from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_deps import (
    GitRemoteInspectorServiceDeps,
)
from src.features.git_remote_inspector.service.git_remote_inspector_service import (
    GitRemoteInspectorDomainService,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_repository_port import (
    GitRemoteInspectorRepositoryPort,
)


class MockRemoteClient(GitRemoteClientPort):
    def fetch_tree(
        self,
        ctx: TraceContext,
        repo: str,
        ref: str,
        recursive: bool = True,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({
            "sha": "tree-sha-123",
            "tree": [
                {"path": "README.md", "type": "blob", "sha": "blob-1", "size": 10},
                {"path": "src/main.py", "type": "blob", "sha": "blob-2", "size": 50},
                {"path": "src/types.py", "type": "blob", "sha": "blob-types", "size": 60},
                {"path": "contracts/openapi.yaml", "type": "blob", "sha": "blob-oas", "size": 150},
                {"path": "src", "type": "tree", "sha": "tree-2"},
            ],
        })

    def fetch_blob(
        self,
        ctx: TraceContext,
        repo: str,
        blob_sha: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"sha": blob_sha, "content": "mock blob"})

    def fetch_file_content(
        self,
        ctx: TraceContext,
        repo: str,
        path: str,
        ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({
            "sha": "blob-2",
            "content": "cHJpbnQoJ2hlbGxvJyk=",
            "encoding": "base64",
            "decoded_content": "print('hello')",
            "size": 14,
        })

    def fetch_diff(
        self,
        ctx: TraceContext,
        repo: str,
        base_ref: str,
        head_ref: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({
            "files": [
                {
                    "filename": "src/main.py",
                    "status": "modified",
                    "patch": "@@ -1 +1 @@\n-old\n+print('hello')",
                }
            ]
        })

    def fetch_branches(
        self,
        ctx: TraceContext,
        repo: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({
            "provider": "github",
            "default_branch": "main",
            "branches": [
                {"name": "main", "commit_sha": "sha-main", "is_default": True, "protected": False},
                {"name": "feature", "commit_sha": "sha-feat", "is_default": False, "protected": False},
            ],
        })


class MockRepository(GitRemoteInspectorRepositoryPort):
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}

    def save_inspection_record(
        self, ctx: TraceContext, record: Dict[str, Any]
    ) -> Result[bool, Exception]:
        self.records[record["id"]] = record
        return Result.ok(True)

    def get_inspection_record(
        self, ctx: TraceContext, record_id: str
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        return Result.ok(self.records.get(record_id))


def test_service_search_files():
    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="test-trace", request_id="req-1", correlation_id="corr-1")
    cmd = SearchFilesCommand(repo_url="owner/repo", ref_name="main", path_pattern="*.py")

    res = service.search_files(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.total_found == 2
    assert any(f.path == "src/main.py" for f in data.files)
    assert any(f.path == "src/types.py" for f in data.files)
    assert data.commit_sha == "tree-sha-123"


def test_service_get_file_content():
    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="test-trace", request_id="req-1", correlation_id="corr-1")
    cmd = GetFileContentCommand(repo_url="owner/repo", file_path="src/main.py", ref_name="main")

    res = service.get_file_content(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.file_path == "src/main.py"
    assert data.content == "print('hello')"


def test_service_get_diff():
    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="test-trace", request_id="req-1", correlation_id="corr-1")
    cmd = GetDiffCommand(repo_url="owner/repo", base_ref="main", head_ref="feature")

    res = service.get_diff(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.files_changed_count == 1
    assert "print('hello')" in data.diff_patch


def test_composite_client_dispatch():
    from src.features.git_remote_inspector.infrastructure.adapters.composite_git_remote_client_adapter import (
        CompositeGitRemoteClientAdapter,
    )
    mock_gh = MockRemoteClient()
    mock_gl = MockRemoteClient()
    mock_bb = MockRemoteClient()
    composite = CompositeGitRemoteClientAdapter(
        github_client=mock_gh,
        gitlab_client=mock_gl,
        bitbucket_client=mock_bb,
    )
    ctx = TraceContext(trace_id="test-trace", request_id="req-1", correlation_id="corr-1")
    assert composite.fetch_tree(ctx, "https://gitlab.com/org/proj", "main").is_ok
    assert composite.fetch_tree(ctx, "https://bitbucket.org/ws/repo", "main").is_ok
    assert composite.fetch_tree(ctx, "https://github.com/org/repo", "main").is_ok


def test_service_error_propagation():
    class FailingClient(GitRemoteClientPort):
        def fetch_tree(self, ctx, repo, ref, recursive=True, token=None):
            from src.features.git_remote_inspector.contracts.git_remote_inspector_types import AuthenticationError
            return Result.err(AuthenticationError("Bad credentials"))
        def fetch_blob(self, ctx, repo, blob_sha, token=None):
            return Result.err(RuntimeError())
        def fetch_file_content(self, ctx, repo, path, ref, token=None):
            return Result.err(RuntimeError())
        def fetch_diff(self, ctx, repo, base_ref, head_ref, token=None):
            return Result.err(RuntimeError())
        def fetch_branches(self, ctx, repo, token=None):
            return Result.err(RuntimeError())

    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=FailingClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-err", request_id="req-err", correlation_id="corr-err")
    cmd = SearchFilesCommand(repo_url="org/repo", ref_name="main")
    res = service.search_files(ctx, cmd)
    assert res.is_err
    assert res.error().status_code == 401


def test_service_discover_contracts():
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        DiscoverContractsCommand,
    )
    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-disc", request_id="req-disc", correlation_id="corr-disc")
    cmd = DiscoverContractsCommand(repo_url="org/repo", ref_name="main")
    res = service.discover_contracts(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.repo_url == "org/repo"
    assert data.primary_language == "python"
    assert data.total_contracts >= 1
    assert any(c.category == "DATA_CONTRACT_TYPES" for c in data.contracts)


def test_service_integrate_context():
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        IntegrateContextCommand,
    )
    deps = GitRemoteInspectorServiceDeps(
        repository=MockRepository(),
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-integ", request_id="req-integ", correlation_id="corr-integ")
    cmd = IntegrateContextCommand(repo_url="org/repo", endpoint_or_entity="search_files", ref_name="main")
    res = service.integrate_context(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.repo_url == "org/repo"
    assert data.target_endpoint_or_entity == "search_files"
    assert data.signature.http_method in ("GET", "POST")
    assert data.signature.path is not None
    assert data.cached is False


def test_service_integrate_context_cache_miss_then_hit():
    from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
        IntegrateContextCommand,
    )
    from src.features.git_remote_inspector.infrastructure.adapters.neo4j_git_remote_inspector_repository_adapter import (
        Neo4jGitRemoteInspectorRepositoryAdapter,
    )
    mock_neo4j = unittest.mock.MagicMock()
    mock_neo4j.query_cypher.return_value = unittest.mock.MagicMock(records=[])

    neo4j_repo = Neo4jGitRemoteInspectorRepositoryAdapter(graph_client=mock_neo4j)
    deps = GitRemoteInspectorServiceDeps(
        repository=neo4j_repo,
        remote_client=MockRemoteClient(),
        config={},
    )
    service = GitRemoteInspectorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-cache", request_id="req-cache", correlation_id="corr-cache")
    cmd = IntegrateContextCommand(repo_url="org/repo", endpoint_or_entity="search_files", ref_name="main")

    first_res = service.integrate_context(ctx, cmd)
    assert first_res.is_ok
    first_data = first_res.unwrap()
    assert first_data.cached is False

    second_res = service.integrate_context(ctx, cmd)
    assert second_res.is_ok
    second_data = second_res.unwrap()
    assert second_data.cached is True
    assert second_data.signature.path == first_data.signature.path


def test_neo4j_adapter_normalized_reconstruction():
    import json
    from src.features.git_remote_inspector.infrastructure.adapters.neo4j_git_remote_inspector_repository_adapter import (
        Neo4jGitRemoteInspectorRepositoryAdapter,
    )
    mock_neo4j = unittest.mock.MagicMock()
    fake_record = {
        "repo_url": "https://github.com/org/service",
        "commit_sha": "abc1234",
        "target_endpoint_or_entity": "/api/v1/auth",
        "http_method": "POST",
        "path": "/api/v1/auth",
        "protocol": "HTTP_REST",
        "summary": "Authenticate user",
        "is_deprecated": False,
        "sample_curl": "curl -X POST https://example.com/api/v1/auth",
        "client_code_snippet": "client.post('/api/v1/auth')",
        "request_body_schema": json.dumps({"type": "object"}),
        "sample_request_payload": json.dumps({"user": "admin"}),
        "response_200_schema": json.dumps({"token": "jwt"}),
        "auth_scheme": "BearerAuth",
        "contract_file": "openapi.yaml",
        "match_type": "OPENAPI_SPEC",
        "related_dto_files": ["dto/auth.ts"],
    }
    mock_neo4j.query_cypher.return_value = unittest.mock.MagicMock(records=[fake_record])

    adapter = Neo4jGitRemoteInspectorRepositoryAdapter(graph_client=mock_neo4j)
    ctx = TraceContext(trace_id="t1", request_id="r1", correlation_id="c1")
    cached = adapter.find_cached_endpoint(ctx, "https://github.com/org/service", "abc1234", "/api/v1/auth")

    assert cached.is_ok
    data = cached.unwrap()
    assert data is not None
    assert data["signature"]["auth_scheme"] == "BearerAuth"
    assert data["signature"]["path"] == "/api/v1/auth"
    assert data["related_dto_files"] == ["dto/auth.ts"]
    assert data["cached"] is True


def test_commit_invariance_preserves_distinct_cache_entries():
    from src.features.git_remote_inspector.infrastructure.adapters.neo4j_git_remote_inspector_repository_adapter import (
        Neo4jGitRemoteInspectorRepositoryAdapter,
    )
    mock_neo4j = unittest.mock.MagicMock()
    mock_neo4j.query_cypher.return_value = unittest.mock.MagicMock(records=[])
    adapter = Neo4jGitRemoteInspectorRepositoryAdapter(graph_client=mock_neo4j)
    ctx = TraceContext(trace_id="t_inv", request_id="r_inv", correlation_id="c_inv")

    commit_a_data = {
        "repo_url": "https://github.com/org/service",
        "target_endpoint_or_entity": "/users",
        "match_type": "OPENAPI_SPEC",
        "contract_file": "openapi.yaml",
        "signature": {
            "http_method": "GET",
            "path": "/users",
            "protocol": "HTTP_REST",
            "summary": "v1 list users",
            "auth_scheme": "BearerAuth",
        },
        "related_dto_files": ["dto/user.ts"],
    }
    commit_b_data = {
        "repo_url": "https://github.com/org/service",
        "target_endpoint_or_entity": "/users",
        "match_type": "OPENAPI_SPEC",
        "contract_file": "openapi.yaml",
        "signature": {
            "http_method": "GET",
            "path": "/users",
            "protocol": "HTTP_REST",
            "summary": "v2 list users with pagination",
            "auth_scheme": "BearerAuth",
        },
        "related_dto_files": ["dto/user_v2.ts"],
    }

    save_a = adapter.save_endpoint_relationship(ctx, "https://github.com/org/service", "sha_commit_1", "main", commit_a_data)
    save_b = adapter.save_endpoint_relationship(ctx, "https://github.com/org/service", "sha_commit_2", "main", commit_b_data)
    assert save_a.is_ok
    assert save_b.is_ok

    hit_a = adapter.find_cached_endpoint(ctx, "https://github.com/org/service", "sha_commit_1", "/users")
    hit_b = adapter.find_cached_endpoint(ctx, "https://github.com/org/service", "sha_commit_2", "/users")

    assert hit_a.is_ok and hit_b.is_ok
    assert hit_a.unwrap()["signature"]["summary"] == "v1 list users"
    assert hit_b.unwrap()["signature"]["summary"] == "v2 list users with pagination"
    assert hit_a.unwrap()["related_dto_files"] == ["dto/user.ts"]
    assert hit_b.unwrap()["related_dto_files"] == ["dto/user_v2.ts"]



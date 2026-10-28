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
    assert data.total_found == 1
    assert data.files[0].path == "src/main.py"
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

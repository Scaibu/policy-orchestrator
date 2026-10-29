import unittest
from typing import Dict, Any, Optional, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
    FileMutationEntry,
    StaleParentCommitError,
    BranchNotFoundError,
    RefUpdateConflictError,
    EmptyMutationBatchError,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_deps import (
    GitRemoteMutatorServiceDeps,
)
from src.features.git_remote_mutator.service.git_remote_mutator_service import (
    GitRemoteMutatorDomainService,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutator_repository_port import (
    GitRemoteMutatorRepositoryPort,
)


class MockMutationClient(GitRemoteMutationClientPort):
    def __init__(self, initial_head_sha: str = "a" * 40) -> None:
        self.head_sha = initial_head_sha
        self.blobs: Dict[str, str] = {}
        self.trees: Dict[str, List[Dict[str, Any]]] = {}
        self.commits: Dict[str, Dict[str, Any]] = {}
        self.refs: Dict[str, str] = {"main": initial_head_sha}

    def get_branch_head(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        if branch not in self.refs:
            return Result.err(BranchNotFoundError(f"Branch {branch} not found"))
        return Result.ok({"branch": branch, "commit_sha": self.refs[branch]})

    def create_blob(
        self,
        ctx: TraceContext,
        repo: str,
        content: str,
        encoding: str = "utf-8",
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        import hashlib
        blob_sha = hashlib.sha1(content.encode("utf-8")).hexdigest()
        self.blobs[blob_sha] = content
        return Result.ok({"sha": blob_sha})

    def create_tree(
        self,
        ctx: TraceContext,
        repo: str,
        base_tree_sha: str,
        tree_nodes: List[Dict[str, Any]],
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        tree_sha = "t" * 40
        self.trees[tree_sha] = tree_nodes
        return Result.ok({"sha": tree_sha, "tree": tree_nodes})

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
        commit_sha = "c" * 40
        self.commits[commit_sha] = {
            "message": message,
            "tree": tree_sha,
            "parents": parent_shas,
        }
        return Result.ok({"sha": commit_sha, "message": message, "tree": tree_sha})

    def update_ref(
        self,
        ctx: TraceContext,
        repo: str,
        branch: str,
        new_commit_sha: str,
        force: bool = False,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        self.refs[branch] = new_commit_sha
        return Result.ok({"ref": f"refs/heads/{branch}", "sha": new_commit_sha})


class MockMutationRepository(GitRemoteMutatorRepositoryPort):
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.branch_pointers: Dict[str, str] = {}

    def save_mutation_record(
        self,
        ctx: TraceContext,
        record: Dict[str, Any],
    ) -> Result[bool, Exception]:
        self.records[record["id"]] = record
        return Result.ok(True)

    def get_mutation_record(
        self,
        ctx: TraceContext,
        record_id: str,
    ) -> Result[Optional[Dict[str, Any]], Exception]:
        return Result.ok(self.records.get(record_id))

    def update_branch_pointer(
        self,
        ctx: TraceContext,
        repo_url: str,
        branch: str,
        parent_sha: str,
        new_commit_sha: str,
        commit_message: str,
    ) -> Result[bool, Exception]:
        self.branch_pointers[f"{repo_url}:{branch}"] = new_commit_sha
        return Result.ok(True)


def test_mutate_single_file_success():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=client,
        config={},
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-mut-1", request_id="req-mut-1", correlation_id="corr-mut-1")

    cmd = MutateFileCommand(
        repo_url="owner/my-service",
        branch="main",
        file_path="src/config.json",
        content='{"key": "new_val"}',
        commit_message="feat: update config",
    )
    res = service.mutate_file(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.repo_url == "owner/my-service"
    assert data.branch == "main"
    assert data.parent_commit_sha == "a" * 40
    assert data.new_commit_sha == "c" * 40
    assert data.files_mutated_count == 1
    assert len(data.blobs_created) == 1
    assert data.blobs_created[0].path == "src/config.json"
    assert repo.branch_pointers.get("owner/my-service:main") == "c" * 40
    assert len(repo.records) == 1


def test_mutate_file_optimistic_concurrency_conflict():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=client,
        config={},
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-mut-2", request_id="req-mut-2", correlation_id="corr-mut-2")

    cmd = MutateFileCommand(
        repo_url="owner/my-service",
        branch="main",
        file_path="src/config.json",
        content='{"key": "new_val"}',
        commit_message="feat: update config",
        expected_parent_sha="b" * 40,
    )
    res = service.mutate_file(ctx, cmd)
    assert res.is_err
    assert isinstance(res.error(), StaleParentCommitError)
    assert res.error().status_code == 409


def test_mutate_batch_multi_file_atomic():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=client,
        config={},
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-mut-3", request_id="req-mut-3", correlation_id="corr-mut-3")

    entries = [
        FileMutationEntry(path="README.md", content="# Hello", is_deleted=False),
        FileMutationEntry(path="src/server.py", content="print('server')", is_deleted=False),
        FileMutationEntry(path="legacy.py", content="", is_deleted=True),
    ]
    cmd = MutateBatchCommand(
        repo_url="owner/my-service",
        branch="main",
        mutations=entries,
        commit_message="chore: batch atomic update and delete",
    )
    res = service.mutate_batch(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.files_mutated_count == 3
    assert len(data.blobs_created) == 2
    assert client.refs["main"] == "c" * 40
    tree_nodes = client.trees["t" * 40]
    assert any(n["path"] == "legacy.py" and n["sha"] is None for n in tree_nodes)
    assert any(n["path"] == "README.md" and n["sha"] is not None for n in tree_nodes)


def test_delete_file_single():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=client,
        config={},
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-mut-4", request_id="req-mut-4", correlation_id="corr-mut-4")

    cmd = DeleteFileCommand(
        repo_url="owner/my-service",
        branch="main",
        file_path="obsolete.txt",
        commit_message="chore: remove obsolete file",
    )
    res = service.delete_file(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.files_mutated_count == 1
    assert len(data.blobs_created) == 0


def test_empty_mutation_batch_fails():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(
        repository=repo,
        mutation_client=client,
        config={},
    )
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-mut-5", request_id="req-mut-5", correlation_id="corr-mut-5")

    cmd = MutateBatchCommand(
        repo_url="owner/my-service",
        branch="main",
        mutations=[],
        commit_message="empty",
    )
    res = service.mutate_batch(ctx, cmd)
    assert res.is_err
    assert isinstance(res.error(), EmptyMutationBatchError)


def test_mutate_noop_detection_skips_commit():
    class NoOpMockClient(MockMutationClient):
        def create_tree(self, ctx, repo, base_tree_sha, tree_nodes, token=None):
            return Result.ok({"sha": base_tree_sha, "base_tree": base_tree_sha, "tree": tree_nodes})

    client = NoOpMockClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(repository=repo, mutation_client=client, config={})
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-noop", request_id="req-noop", correlation_id="corr-noop")

    cmd = MutateFileCommand(
        repo_url="owner/my-service",
        branch="main",
        file_path="README.md",
        content="unchanged content",
        commit_message="chore: no-op commit",
    )
    res = service.mutate_file(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.is_noop is True
    assert data.files_mutated_count == 0
    assert data.new_commit_sha == "a" * 40
    assert client.refs["main"] == "a" * 40


def test_file_mode_inference_executable():
    from src.features.git_remote_mutator.domain.rules.git_remote_mutator_rules import (
        infer_git_file_mode,
        calculate_git_blob_sha,
    )
    assert infer_git_file_mode("deploy.sh", "echo 1") == "100755"
    assert infer_git_file_mode("bin/run.bash", "echo 1") == "100755"
    assert infer_git_file_mode("scripts/task", "#!/usr/bin/env python3\nprint(1)") == "100755"
    assert infer_git_file_mode("docs/README.md", "# Hello") == "100644"
    assert infer_git_file_mode("arbitrary.txt", "# Hello", explicit_mode="100755") == "100755"

    sha_text = calculate_git_blob_sha("hello world\n", encoding="utf-8")
    assert isinstance(sha_text, str)
    assert len(sha_text) == 40


def test_mutate_file_base64_encoding():
    client = MockMutationClient(initial_head_sha="a" * 40)
    repo = MockMutationRepository()
    deps = GitRemoteMutatorServiceDeps(repository=repo, mutation_client=client, config={})
    service = GitRemoteMutatorDomainService(deps=deps)
    ctx = TraceContext(trace_id="trace-b64", request_id="req-b64", correlation_id="corr-b64")

    import base64
    b64_content = base64.b64encode(b"\x00\x01\x02\x03\x04\xff").decode("ascii")
    cmd = MutateFileCommand(
        repo_url="owner/my-service",
        branch="main",
        file_path="assets/binary.bin",
        content=b64_content,
        encoding="base64",
        commit_message="feat: upload binary asset via base64",
    )
    res = service.mutate_file(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.is_noop is False
    assert len(data.blobs_created) == 1
    assert data.blobs_created[0].size == len(b64_content)

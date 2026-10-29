from typing import Dict, Any, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    BranchNotFoundError,
    BranchAlreadyExistsError,
    MergeConflictError,
    InvalidBranchNameError,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_deps import (
    GitBranchLifecycleServiceDeps,
)
from src.features.git_branch_lifecycle.service.git_branch_lifecycle_service import (
    GitBranchLifecycleDomainService,
)
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_client_port import (
    GitBranchLifecycleClientPort,
)
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_repository_port import (
    GitBranchLifecycleRepositoryPort,
)
from src.features.git_branch_lifecycle.domain.rules.git_branch_lifecycle_rules import (
    normalize_branch_name,
)


class MockLifecycleClient(GitBranchLifecycleClientPort):
    def __init__(self, initial_head_sha: str = "a" * 40) -> None:
        self.branches: Dict[str, str] = {"main": initial_head_sha}
        self.repos: Dict[str, Dict[str, Any]] = {}

    def resolve_ref_sha(
        self, ctx: TraceContext, repo: str, ref: str, token: Optional[str] = None
    ) -> Result[str, Exception]:
        clean = normalize_branch_name(ref)
        if clean in self.branches:
            return Result.ok(self.branches[clean])
        return Result.err(BranchNotFoundError(f"Branch {ref} not found"))

    def create_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, target_sha: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        if branch in self.branches:
            return Result.err(BranchAlreadyExistsError(f"Branch {branch} already exists"))
        self.branches[branch] = target_sha
        return Result.ok({"ref": f"refs/heads/{branch}", "object": {"sha": target_sha}})

    def delete_branch_ref(
        self, ctx: TraceContext, repo: str, branch: str, token: Optional[str] = None
    ) -> Result[bool, Exception]:
        if branch not in self.branches:
            return Result.err(BranchNotFoundError(f"Branch {branch} not found"))
        del self.branches[branch]
        return Result.ok(True)

    def merge_branches(
        self,
        ctx: TraceContext,
        repo: str,
        base: str,
        head: str,
        commit_message: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Result[Dict[str, Any], Exception]:
        if base not in self.branches or head not in self.branches:
            return Result.err(BranchNotFoundError("Base or head branch not found"))
        if self.branches[base] == self.branches[head]:
            return Result.ok({"sha": self.branches[base], "message": "Already up to date"})
        merge_sha = "m" * 40
        self.branches[base] = merge_sha
        return Result.ok({"sha": merge_sha, "message": commit_message or "Merged"})

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
        full_name = f"{org}/{name}" if org else f"user/{name}"
        if full_name in self.repos:
            return Result.err(RuntimeError(f"Repository {full_name} already exists"))
        record = {
            "name": name,
            "html_url": f"https://github.com/{full_name}",
            "private": is_private,
        }
        self.repos[full_name] = record
        return Result.ok(record)


class MockLifecycleRepository(GitBranchLifecycleRepositoryPort):
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.graph_branches: Dict[str, str] = {}

    def save_lifecycle_record(
        self, ctx: TraceContext, record: Dict[str, Any]
    ) -> Result[bool, Exception]:
        self.records[str(record.get("id"))] = record
        return Result.ok(True)

    def create_branch_node(
        self, ctx: TraceContext, repo: str, branch: str, commit_sha: str
    ) -> Result[bool, Exception]:
        self.graph_branches[f"{repo}:{branch}"] = commit_sha
        return Result.ok(True)

    def delete_branch_node(
        self, ctx: TraceContext, repo: str, branch: str
    ) -> Result[bool, Exception]:
        key = f"{repo}:{branch}"
        if key in self.graph_branches:
            del self.graph_branches[key]
        return Result.ok(True)


def test_create_branch_success():
    client = MockLifecycleClient(initial_head_sha="a" * 40)
    repo = MockLifecycleRepository()
    deps = GitBranchLifecycleServiceDeps(repository=repo, client=client, config={})
    service = GitBranchLifecycleDomainService(deps=deps)
    ctx = TraceContext(trace_id="t-1", request_id="r-1", correlation_id="c-1")

    cmd = CreateBranchCommand(
        repo_url="owner/repo",
        branch_name="feature/new-login",
        source_ref="main",
    )
    res = service.create_branch(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.action == "CREATE_BRANCH"
    assert data.branch_name == "feature/new-login"
    assert data.result_sha == "a" * 40
    assert "feature/new-login" in client.branches
    assert len(repo.records) == 1
    assert "owner/repo:feature/new-login" in repo.graph_branches


def test_create_branch_invalid_characters_fails():
    client = MockLifecycleClient()
    repo = MockLifecycleRepository()
    deps = GitBranchLifecycleServiceDeps(repository=repo, client=client, config={})
    service = GitBranchLifecycleDomainService(deps=deps)
    ctx = TraceContext(trace_id="t-2", request_id="r-2", correlation_id="c-2")

    cmd = CreateBranchCommand(
        repo_url="owner/repo",
        branch_name="feature/invalid..branch",
        source_ref="main",
    )
    res = service.create_branch(ctx, cmd)
    assert res.is_err
    assert isinstance(res.error(), InvalidBranchNameError)


def test_delete_branch_success():
    client = MockLifecycleClient(initial_head_sha="a" * 40)
    client.branches["feature/obsolete"] = "b" * 40
    repo = MockLifecycleRepository()
    repo.graph_branches["owner/repo:feature/obsolete"] = "b" * 40
    deps = GitBranchLifecycleServiceDeps(repository=repo, client=client, config={})
    service = GitBranchLifecycleDomainService(deps=deps)
    ctx = TraceContext(trace_id="t-3", request_id="r-3", correlation_id="c-3")

    cmd = DeleteBranchCommand(
        repo_url="owner/repo",
        branch_name="feature/obsolete",
    )
    res = service.delete_branch(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.action == "DELETE_BRANCH"
    assert "feature/obsolete" not in client.branches
    assert "owner/repo:feature/obsolete" not in repo.graph_branches


def test_merge_branch_success():
    client = MockLifecycleClient(initial_head_sha="a" * 40)
    client.branches["feature/patch"] = "p" * 40
    repo = MockLifecycleRepository()
    deps = GitBranchLifecycleServiceDeps(repository=repo, client=client, config={})
    service = GitBranchLifecycleDomainService(deps=deps)
    ctx = TraceContext(trace_id="t-4", request_id="r-4", correlation_id="c-4")

    cmd = MergeBranchCommand(
        repo_url="owner/repo",
        base_branch="main",
        head_branch="feature/patch",
        commit_message="Merge pull request #1 from feature/patch",
    )
    res = service.merge_branch(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.action == "MERGE_BRANCH"
    assert data.result_sha == "m" * 40
    assert client.branches["main"] == "m" * 40


def test_create_repository_success():
    client = MockLifecycleClient()
    repo = MockLifecycleRepository()
    deps = GitBranchLifecycleServiceDeps(repository=repo, client=client, config={})
    service = GitBranchLifecycleDomainService(deps=deps)
    ctx = TraceContext(trace_id="t-5", request_id="r-5", correlation_id="c-5")

    cmd = CreateRepositoryCommand(
        repo_name="my-brand-new-repo",
        description="Managed via Policy Orchestrator",
        is_private=True,
    )
    res = service.create_repository(ctx, cmd)
    assert res.is_ok
    data = res.unwrap()
    assert data.action == "CREATE_REPO"
    assert "my-brand-new-repo" in data.repo_url

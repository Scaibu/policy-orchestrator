import unittest
from typing import Dict, Any, Optional, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
    SubmoduleNotFoundError,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_deps import (
    GitSubmoduleManagerServiceDeps,
)
from src.features.git_submodule_manager.service.git_submodule_manager_service import (
    GitSubmoduleManagerDomainService,
)
from src.features.git_submodule_manager.infrastructure.ports.git_submodule_manager_repository_port import (
    GitSubmoduleManagerRepositoryPort,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutation_client_port import (
    GitRemoteMutationClientPort,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_client_port import (
    GitRemoteClientPort,
)


class MockMutationClient(GitRemoteMutationClientPort):
    def __init__(self, initial_head_sha: str = "a" * 40) -> None:
        self.head_sha = initial_head_sha
        self.blobs: Dict[str, str] = {}
        self.trees: Dict[str, List[Dict[str, Any]]] = {}
        self.commits: Dict[str, Dict[str, Any]] = {}
        self.refs: Dict[str, str] = {"main": initial_head_sha}

    def get_branch_head(
        self, ctx: TraceContext, repo: str, branch: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"branch": branch, "commit_sha": self.refs.get(branch, self.head_sha)})

    def create_blob(
        self, ctx: TraceContext, repo: str, content: str, encoding: str = "utf-8", token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        import hashlib
        blob_sha = hashlib.sha1(content.encode("utf-8")).hexdigest()
        self.blobs[blob_sha] = content
        return Result.ok({"sha": blob_sha})

    def create_tree(
        self, ctx: TraceContext, repo: str, base_tree_sha: str, tree_nodes: List[Dict[str, Any]], token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        import hashlib
        tree_sha = hashlib.sha1(str(tree_nodes).encode("utf-8")).hexdigest()
        self.trees[tree_sha] = tree_nodes
        return Result.ok({"sha": tree_sha})

    def create_commit(
        self, ctx: TraceContext, repo: str, message: str, tree_sha: str, parent_shas: List[str], author_name: Optional[str] = None, author_email: Optional[str] = None, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        import hashlib
        commit_sha = hashlib.sha1(f"{message}{tree_sha}".encode("utf-8")).hexdigest()
        self.commits[commit_sha] = {"message": message, "tree": tree_sha, "parents": parent_shas}
        return Result.ok({"sha": commit_sha})

    def update_ref(
        self, ctx: TraceContext, repo: str, branch: str, new_commit_sha: str, force: bool = False, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        self.refs[branch] = new_commit_sha
        return Result.ok({"ref": f"refs/heads/{branch}", "sha": new_commit_sha})


class MockInspectorClient(GitRemoteClientPort):
    def __init__(self) -> None:
        self.files: Dict[str, str] = {}

    def fetch_file_content(
        self, ctx: TraceContext, repo: str, path: str, ref: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        if path in self.files:
            return Result.ok({"content": self.files[path], "encoding": "utf-8"})
        return Result.err(FileNotFoundError(path))

    def fetch_tree(
        self, ctx: TraceContext, repo: str, ref: str, recursive: bool = True, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"tree": [{"path": "vendor/lib", "mode": "160000", "sha": "c" * 40}]})

    def fetch_blob(
        self, ctx: TraceContext, repo: str, blob_sha: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"content": ""})

    def fetch_branches(
        self, ctx: TraceContext, repo: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"branches": []})

    def fetch_diff(
        self, ctx: TraceContext, repo: str, base_ref: str, head_ref: str, token: Optional[str] = None
    ) -> Result[Dict[str, Any], Exception]:
        return Result.ok({"diff": ""})


class MockSubmoduleRepository(GitSubmoduleManagerRepositoryPort):
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.graph_nodes: List[Dict[str, Any]] = []

    def save_submodule_record(self, ctx: TraceContext, record: Dict[str, Any]) -> Result[bool, Exception]:
        self.records.append(record)
        return Result.ok(True)

    def get_submodule_record(self, ctx: TraceContext, record_id: str) -> Result[Optional[Dict[str, Any]], Exception]:
        for r in self.records:
            if r["id"] == record_id:
                return Result.ok(r)
        return Result.ok(None)

    def sync_submodule_graph(self, ctx: TraceContext, parent_repo: str, submodule_path: str, submodule_url: str, gitlink_sha: str) -> Result[bool, Exception]:
        self.graph_nodes.append({
            "parent_repo": parent_repo,
            "path": submodule_path,
            "url": submodule_url,
            "sha": gitlink_sha,
        })
        return Result.ok(True)


class TestGitSubmoduleManagerService(unittest.TestCase):
    def setUp(self):
        self.repo = MockSubmoduleRepository()
        self.mutator_client = MockMutationClient(initial_head_sha="1" * 40)
        self.inspector_client = MockInspectorClient()
        self.deps = GitSubmoduleManagerServiceDeps(
            repository=self.repo,
            mutator_client=self.mutator_client,
            inspector_client=self.inspector_client,
        )
        self.service = GitSubmoduleManagerDomainService(deps=self.deps)
        self.ctx = TraceContext(trace_id="test-trace", request_id="test-req", correlation_id="test-corr")

    def test_add_submodule(self):
        cmd = AddSubmoduleCommand(
            repo_url="owner/parent",
            branch="main",
            submodule_path="vendor/dep",
            submodule_url="owner/dep",
            submodule_ref="2" * 40,
            commit_message="feat: add dep submodule",
        )
        res = self.service.add_submodule(self.ctx, cmd)
        self.assertTrue(res.is_ok)
        out = res.unwrap()
        self.assertEqual(out.action, "ADD_SUBMODULE")
        self.assertIn("vendor/dep", out.submodules_affected)
        self.assertEqual(len(self.repo.records), 1)
        self.assertEqual(len(self.repo.graph_nodes), 1)

    def test_update_submodule_not_found(self):
        cmd = UpdateSubmoduleCommand(
            repo_url="owner/parent",
            branch="main",
            submodule_path="vendor/nonexistent",
            target_commit_sha="3" * 40,
        )
        res = self.service.update_submodule(self.ctx, cmd)
        self.assertTrue(res.is_err)
        self.assertIsInstance(res.error(), SubmoduleNotFoundError)

    def test_pull_submodules(self):
        self.inspector_client.files[".gitmodules"] = (
            '[submodule "vendor/lib"]\n\tpath = vendor/lib\n\turl = owner/lib\n\tbranch = main\n'
        )
        cmd = PullSubmodulesCommand(
            repo_url="owner/parent",
            branch="main",
            submodule_paths=["vendor/lib"],
        )
        res = self.service.pull_submodules(self.ctx, cmd)
        self.assertTrue(res.is_ok)
        out = res.unwrap()
        self.assertEqual(out.action, "PULL_SUBMODULES")

    def test_push_multi_repo(self):
        self.inspector_client.files[".gitmodules"] = (
            '[submodule "vendor/lib"]\n\tpath = vendor/lib\n\turl = owner/lib\n\tbranch = main\n'
        )
        cmd = PushMultiRepoCommand(
            parent_repo_url="owner/parent",
            parent_branch="main",
            submodule_paths=["vendor/lib"],
            verify_remote_exists=True,
        )
        res = self.service.push_multi_repo(self.ctx, cmd)
        self.assertTrue(res.is_ok)
        out = res.unwrap()
        self.assertEqual(out.action, "PUSH_MULTI_REPO")

    def test_get_sync_status(self):
        self.inspector_client.files[".gitmodules"] = (
            '[submodule "vendor/lib"]\n\tpath = vendor/lib\n\turl = owner/lib\n\tbranch = main\n'
        )
        cmd = SubmoduleSyncStatusCommand(
            repo_url="owner/parent",
            branch="main",
        )
        res = self.service.get_sync_status(self.ctx, cmd)
        self.assertTrue(res.is_ok)
        out = res.unwrap()
        self.assertEqual(out.action, "SYNC_STATUS")
        self.assertIsNotNone(out.status_details)
        self.assertEqual(len(out.status_details), 1)


if __name__ == "__main__":
    unittest.main()

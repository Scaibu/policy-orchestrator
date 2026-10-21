"""
================================================================================
BLUEPRINT: GIT VERIFIER DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW:
   Coordinates Git repository and submodule verification:
   - Resolves active branch and commit hash via HEAD inspection.
   - Cross-references .gitmodules declarations with child repository commits.
   - Detects synchronization drift and records audit metrics to database.
================================================================================
"""
import time
import uuid
from pathlib import Path
from typing import List, Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.git_verifier.contracts.git_verifier_types import (
    VerifyGitCommand,
    GitVerificationResult,
    SubmoduleStatus,
    WorkingTreeStatus,
    RemoteTrackingStatus,
)
from src.features.git_verifier.domain.rules.git_verifier_rules import (
    read_git_head,
    parse_gitmodules,
    resolve_submodule_commit,
    get_gitlink_expected_commit,
    inspect_working_tree,
    inspect_remote_tracking,
)
from src.features.git_verifier.domain.machines.git_verifier_machine import (
    STATE_PENDING,
    STATE_READING_HEAD,
    STATE_INSPECTING,
    STATE_COMPLETED,
    transition_to,
)
from src.features.git_verifier.infrastructure.ports.git_verifier_repository_port import (
    GitVerifierRepositoryPort,
)


class GitVerifierDomainService(AbstractTracedComponent):
    def __init__(self, repository: GitVerifierRepositoryPort) -> None:
        super().__init__()
        self._repository = repository

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def verify_git(
        self, ctx: TraceContext, cmd: VerifyGitCommand
    ) -> Result[GitVerificationResult, Exception]:
        start = time.perf_counter()
        span = None
        if self._tracer:
            span = self._tracer.start_span("git_verifier.verify_git")

        try:
            repo_root = Path(cmd.repo_path).resolve()
            if not repo_root.exists():
                raise FileNotFoundError(f"Git repository path does not exist: {cmd.repo_path}")

            git_dir = repo_root / ".git"
            if not git_dir.exists():
                raise ValueError(f"Path is not a git repository (missing .git): {cmd.repo_path}")

            state = transition_to(STATE_PENDING, STATE_READING_HEAD)
            real_git_dir = git_dir
            if git_dir.is_file():
                content = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
                if content.startswith("gitdir:"):
                    real_git_dir = (repo_root / content.split(":", 1)[1].strip()).resolve()

            branch, head_commit, is_detached = read_git_head(real_git_dir)

            state = transition_to(state, STATE_INSPECTING)
            upstream, ahead, behind, is_in_sync = inspect_remote_tracking(repo_root, branch)
            remote_status = RemoteTrackingStatus(
                upstream_branch=upstream,
                ahead_count=ahead,
                behind_count=behind,
                is_in_sync=is_in_sync,
            )

            staged, unstaged, untracked, is_wt_clean = ([], [], [], True)
            if cmd.check_working_tree:
                staged, unstaged, untracked, is_wt_clean = inspect_working_tree(repo_root)

            working_tree_status = WorkingTreeStatus(
                staged_files=staged,
                unstaged_files=unstaged,
                untracked_files=untracked,
                is_clean=is_wt_clean,
            )

            submodules_status: List[SubmoduleStatus] = []
            has_mismatch = False

            if cmd.check_submodules:
                declared_submodules = parse_gitmodules(repo_root)
                for sub in declared_submodules:
                    actual_commit, is_sub_dirty = resolve_submodule_commit(repo_root, sub["path"])
                    expected_gitlink = get_gitlink_expected_commit(repo_root, sub["path"])
                    is_synced = (actual_commit is not None and (expected_gitlink is None or expected_gitlink == actual_commit))
                    if not is_synced or is_sub_dirty:
                        has_mismatch = True
                    submodules_status.append(
                        SubmoduleStatus(
                            name=sub["name"],
                            path=sub["path"],
                            expected_commit=expected_gitlink,
                            actual_commit=actual_commit,
                            is_synchronized=is_synced,
                            is_dirty=is_sub_dirty,
                        )
                    )

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start) * 1000

            overall_clean = is_wt_clean and not has_mismatch

            record_id = f"gitver_{uuid.uuid4().hex[:12]}"
            self._repository.record_verification(
                record_id=record_id,
                repo_path=str(repo_root),
                is_clean=overall_clean,
                submodules_count=len(submodules_status),
                duration_ms=duration_ms,
            )

            result = GitVerificationResult(
                repo_path=str(repo_root),
                current_branch=branch,
                head_commit=head_commit,
                is_detached=is_detached,
                is_clean=overall_clean,
                remote=remote_status,
                working_tree=working_tree_status,
                submodules=submodules_status,
                has_submodule_mismatch=has_mismatch,
                duration_ms=round(duration_ms, 2),
                status=state,
            )
            return Result.ok(result)

        except Exception as exc:
            mapped = self.mapFailure(exc)
            return Result.err(mapped)
        finally:
            if span:
                span.end()

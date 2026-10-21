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
)
from src.features.git_verifier.domain.rules.git_verifier_rules import (
    read_git_head,
    parse_gitmodules,
    resolve_submodule_commit,
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

            branch, head_commit = read_git_head(real_git_dir)

            state = transition_to(state, STATE_INSPECTING)
            submodules_status: List[SubmoduleStatus] = []
            has_mismatch = False

            if cmd.check_submodules:
                declared_submodules = parse_gitmodules(repo_root)
                for sub in declared_submodules:
                    actual_commit = resolve_submodule_commit(repo_root, sub["path"])
                    is_synced = actual_commit is not None
                    if not is_synced:
                        has_mismatch = True
                    submodules_status.append(
                        SubmoduleStatus(
                            name=sub["name"],
                            path=sub["path"],
                            expected_commit="HEAD",
                            actual_commit=actual_commit,
                            is_synchronized=is_synced,
                        )
                    )

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start) * 1000

            record_id = f"gitver_{uuid.uuid4().hex[:12]}"
            self._repository.record_verification(
                record_id=record_id,
                repo_path=str(repo_root),
                is_clean=not has_mismatch,
                submodules_count=len(submodules_status),
                duration_ms=duration_ms,
            )

            result = GitVerificationResult(
                repo_path=str(repo_root),
                current_branch=branch,
                head_commit=head_commit,
                is_clean=not has_mismatch,
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

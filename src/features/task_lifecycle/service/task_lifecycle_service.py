"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: TASK LIFECYCLE DOMAIN SERVICE
================================================================================
Pure domain orchestrator coordinating issue creation, branch checkouts, verification
test gates, PR merges, and parent submodule commit pointer synchronization.

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

import os
from typing import Optional
from src.features.task_lifecycle.contracts.task_lifecycle_types import (
    StartTaskCommand,
    StartTaskResult,
    FinishTaskCommand,
    FinishTaskResult,
    SyncPointerCommand,
    SyncPointerResult,
)
from src.features.task_lifecycle.infrastructure.ports.task_lifecycle_port import GitTaskExecutionPort


class TaskLifecycleDomainService:

    def __init__(self, port: GitTaskExecutionPort):
        self._port = port

    def start_task(self, cmd: StartTaskCommand) -> StartTaskResult:
        body = cmd.body or f"Automated feature task for: {cmd.title}"
        issue_num, issue_url = self._port.create_github_issue(cmd.repo, cmd.title, body)
        branch_name = f"feat/{issue_num}-{cmd.slug}"
        self._port.create_and_checkout_branch(branch_name, cmd.base_branch)
        return StartTaskResult(
            repo=cmd.repo,
            issue_number=issue_num,
            issue_url=issue_url,
            branch_name=branch_name,
            status="SUCCESS",
        )

    def finish_task(self, cmd: FinishTaskCommand, current_branch: str) -> FinishTaskResult:
        if cmd.test_cmd:
            tests_passed = self._port.run_verification_tests(cmd.test_cmd)
            if not tests_passed:
                raise RuntimeError(f"Verification tests failed for command: {cmd.test_cmd}")

        pr_title = cmd.pr_title or f"feat: resolve issue #{cmd.issue_number}"
        pr_body = cmd.pr_body or f"Automated completion of task. Closes #{cmd.issue_number}"

        pr_number, pr_url = self._port.create_pull_request(
            repo=cmd.repo,
            base=cmd.base_branch,
            head=current_branch,
            title=pr_title,
            body=pr_body,
        )

        merged = self._port.merge_pull_request(cmd.repo, pr_number, delete_branch=True)
        self._port.checkout_and_pull(cmd.base_branch)
        self._port.close_github_issue(cmd.repo, cmd.issue_number, "Resolved by PR merge.")

        parent_synced = False
        if cmd.sync_parent_pointer:
            submodule_name = os.path.basename(os.getcwd())
            parent_dir = os.path.dirname(os.getcwd())
            if os.path.exists(os.path.join(parent_dir, ".gitmodules")):
                sync_msg = f"chore(submodule): update {submodule_name} pointer to resolve #{cmd.issue_number}"
                res = self._port.sync_parent_submodule(submodule_name, parent_dir, sync_msg)
                parent_synced = res is not None

        return FinishTaskResult(
            repo=cmd.repo,
            issue_number=cmd.issue_number,
            pr_number=pr_number,
            pr_url=pr_url,
            merged=merged,
            branch_deleted=True,
            parent_synced=parent_synced,
            status="SUCCESS",
        )

    def sync_pointer(self, cmd: SyncPointerCommand) -> SyncPointerResult:
        commit_msg = cmd.commit_msg or f"chore(submodule): update {cmd.submodule_path} pointer"
        parent_rev = self._port.sync_parent_submodule(cmd.submodule_path, cmd.parent_repo_path, commit_msg)
        return SyncPointerResult(
            submodule_path=cmd.submodule_path,
            commit_hash="HEAD",
            parent_commit=parent_rev or "UNCHANGED",
            status="SUCCESS" if parent_rev else "FAILED",
        )


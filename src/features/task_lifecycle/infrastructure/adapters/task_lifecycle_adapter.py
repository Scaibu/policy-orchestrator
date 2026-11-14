"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: TASK LIFECYCLE CLI & PROCESS ADAPTER
================================================================================
Implements GitTaskExecutionPort by spawning subprocess invocations of `git` and `gh`.
Wraps execution in resilient error checking with zero inline comments in method bodies.

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

import os
import re
import subprocess
from typing import Optional, Tuple
from src.features.task_lifecycle.infrastructure.ports.task_lifecycle_port import GitTaskExecutionPort


class SubprocessGitTaskAdapter(GitTaskExecutionPort):

    def __init__(self, working_dir: str = "."):
        self._cwd = os.path.abspath(working_dir)

    def create_github_issue(self, repo: str, title: str, body: str) -> Tuple[int, str]:
        cmd = ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", body]
        res = subprocess.run(cmd, cwd=self._cwd, text=True, capture_output=True, check=True)
        out = res.stdout.strip()
        match = re.search(r"/issues/(\d+)", out)
        if not match:
            raise RuntimeError(f"Failed to parse issue number from gh output: {out}")
        issue_num = int(match.group(1))
        return issue_num, out

    def create_and_checkout_branch(self, branch_name: str, base_branch: str) -> None:
        subprocess.run(["git", "checkout", base_branch], cwd=self._cwd, check=True, capture_output=True)
        subprocess.run(["git", "pull", "origin", base_branch], cwd=self._cwd, check=True, capture_output=True)
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=self._cwd, check=True, capture_output=True)

    def run_verification_tests(self, test_cmd: str) -> bool:
        res = subprocess.run(test_cmd, shell=True, cwd=self._cwd)
        return res.returncode == 0

    def create_pull_request(self, repo: str, base: str, head: str, title: str, body: str) -> Tuple[int, str]:
        subprocess.run(["git", "push", "-u", "origin", head], cwd=self._cwd, check=True, capture_output=True)
        cmd = ["gh", "pr", "create", "--repo", repo, "--base", base, "--head", head, "--title", title, "--body", body]
        res = subprocess.run(cmd, cwd=self._cwd, text=True, capture_output=True, check=True)
        out = res.stdout.strip()
        match = re.search(r"/pull/(\d+)", out)
        if not match:
            raise RuntimeError(f"Failed to parse pull request number from gh output: {out}")
        pr_num = int(match.group(1))
        return pr_num, out

    def merge_pull_request(self, repo: str, pr_number: int, delete_branch: bool) -> bool:
        cmd = ["gh", "pr", "merge", "--repo", repo, str(pr_number), "--merge"]
        if delete_branch:
            cmd.append("--delete-branch")
        res = subprocess.run(cmd, cwd=self._cwd, text=True, capture_output=True, check=True)
        return res.returncode == 0

    def checkout_and_pull(self, branch: str) -> None:
        subprocess.run(["git", "checkout", branch], cwd=self._cwd, check=True, capture_output=True)
        subprocess.run(["git", "pull", "origin", branch], cwd=self._cwd, check=True, capture_output=True)

    def close_github_issue(self, repo: str, issue_number: int, reason: str) -> None:
        subprocess.run(["gh", "issue", "close", "--repo", repo, str(issue_number)], cwd=self._cwd, capture_output=True)

    def sync_parent_submodule(self, submodule_path: str, parent_path: str, commit_msg: str) -> Optional[str]:
        p_dir = os.path.abspath(parent_path)
        subprocess.run(["git", "add", submodule_path], cwd=p_dir, check=True, capture_output=True)
        res = subprocess.run(["git", "commit", "-m", commit_msg], cwd=p_dir, capture_output=True, text=True)
        if res.returncode == 0:
            subprocess.run(["git", "push", "origin", "main"], cwd=p_dir, check=True, capture_output=True)
            head_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=p_dir, text=True).strip()
            return head_rev
        return None


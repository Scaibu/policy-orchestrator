"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: UNIFIED TASK LIFECYCLE CLI COMMAND
================================================================================
Provides a single unified command (`task`) taking parameters to drive the entire
development lifecycle:
  - Phase 'start': Creates GitHub issue, cuts feature branch, and optionally scaffolds feature files.
  - Phase 'finish': Runs test verification gate, creates PR, merges, closes issue, and syncs parent pointer.
  - Phase 'scaffold': Executes hexagonal scaffolding for a feature in a target package.
  - Phase 'sync': Updates submodule commit pointer in parent repository.

ZERO-INLINE-COMMENT DOCTRINE: Method bodies remain 100% comment-free and pure.
================================================================================
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from src.features.task_lifecycle.contracts.task_lifecycle_types import (
    StartTaskCommand,
    FinishTaskCommand,
    SyncPointerCommand,
)
from src.features.task_lifecycle.infrastructure.adapters.task_lifecycle_adapter import (
    SubprocessGitTaskAdapter,
)
from src.features.task_lifecycle.service.task_lifecycle_service import (
    TaskLifecycleDomainService,
)


def _get_current_repo_remote(cwd: str = ".") -> str:
    try:
        url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=cwd, text=True).strip()
        if "github.com:" in url:
            return url.split("github.com:")[1].replace(".git", "")
        elif "github.com/" in url:
            return url.split("github.com/")[1].replace(".git", "")
        return url
    except Exception:
        return ""


def _get_current_branch(cwd: str = ".") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return ""


def _execute_scaffold_if_requested(args: argparse.Namespace) -> None:
    package_name = getattr(args, "package", None)
    feature_name = getattr(args, "feature", None)
    if package_name and feature_name:
        base_dir = getattr(args, "base_dir", ".")
        cli_path = Path(__file__).resolve().parent.parent.parent / "cli" / "folder_scaffold_cli.py"
        if cli_path.exists():
            cmd = [
                sys.executable,
                str(cli_path),
                "scaffold-feature",
                "--package",
                package_name,
                "--name",
                feature_name,
                "--base-dir",
                base_dir,
            ]
            subprocess.run(cmd, check=True)
            print(f"Scaffolded 14-role hexagonal feature: {package_name}/src/features/{feature_name}")


def handle_unified_task_command(args: argparse.Namespace) -> int:
    phase = getattr(args, "phase", "start")
    repo = getattr(args, "repo", None) or _get_current_repo_remote()

    port = SubprocessGitTaskAdapter()
    service = TaskLifecycleDomainService(port=port)

    if phase == "start":
        title = getattr(args, "title", None)
        slug = getattr(args, "slug", None) or getattr(args, "feature", None)
        if not title or not slug:
            print("Error: --title and (--slug or --feature) are required for phase 'start'.")
            return 1
        if not repo:
            print("Error: --repo is required or remote.origin.url must be configured.")
            return 1

        cmd = StartTaskCommand(
            repo=repo,
            title=title,
            slug=slug,
            base_branch=getattr(args, "base_branch", "main"),
            body=getattr(args, "body", None),
        )
        try:
            res = service.start_task(cmd)
            _execute_scaffold_if_requested(args)
            if getattr(args, "json", False):
                print(json.dumps(res.__dict__, indent=2))
            else:
                print("=" * 80)
                print("Task Started Successfully")
                print(f"Repository:   {res.repo}")
                print(f"Issue Number: #{res.issue_number}")
                print(f"Issue URL:    {res.issue_url}")
                print(f"Branch:       {res.branch_name}")
                print("=" * 80)
            return 0
        except Exception as exc:
            print(f"Error starting task: {exc}")
            return 1

    elif phase == "finish":
        if not repo:
            print("Error: --repo is required or remote.origin.url must be configured.")
            return 1

        current_branch = _get_current_branch()
        if current_branch in ("main", "master"):
            print("Error: cannot finish task while on main/master branch. Must be on feature branch.")
            return 1

        issue_num = getattr(args, "issue", None)
        if not issue_num:
            import re
            m = re.search(r"feat/(\d+)-", current_branch)
            if m:
                issue_num = int(m.group(1))
            else:
                print("Error: --issue is required when branch name does not match feat/<issue>-<slug> pattern.")
                return 1

        cmd = FinishTaskCommand(
            repo=repo,
            issue_number=int(issue_num),
            pr_title=getattr(args, "title", None),
            pr_body=getattr(args, "body", None),
            base_branch=getattr(args, "base_branch", "main"),
            test_cmd=getattr(args, "test_cmd", None),
            sync_parent_pointer=not getattr(args, "no_sync", False),
        )

        try:
            res = service.finish_task(cmd, current_branch=current_branch)
            if getattr(args, "json", False):
                print(json.dumps(res.__dict__, indent=2))
            else:
                print("=" * 80)
                print("Task Finished and Merged Successfully")
                print(f"Repository:     {res.repo}")
                print(f"Issue Closed:   #{res.issue_number}")
                print(f"Pull Request:   #{res.pr_number} ({res.pr_url})")
                print(f"Branch Deleted: {res.branch_deleted}")
                print(f"Parent Synced:  {res.parent_synced}")
                print("=" * 80)
            return 0
        except Exception as exc:
            print(f"Error finishing task: {exc}")
            return 1

    elif phase == "scaffold":
        _execute_scaffold_if_requested(args)
        return 0

    elif phase == "sync":
        submodule = getattr(args, "submodule", None) or os.path.basename(os.getcwd())
        parent_path = getattr(args, "parent_path", "..")
        cmd = SyncPointerCommand(
            submodule_path=submodule,
            parent_repo_path=parent_path,
            commit_msg=getattr(args, "message", None),
        )
        try:
            res = service.sync_pointer(cmd)
            if getattr(args, "json", False):
                print(json.dumps(res.__dict__, indent=2))
            else:
                print(f"Submodule '{res.submodule_path}' pointer synced: {res.parent_commit}")
            return 0
        except Exception as exc:
            print(f"Error syncing pointer: {exc}")
            return 1

    else:
        print(f"Unknown phase: {phase}. Choices are start, finish, scaffold, sync.")
        return 1


def register_task_lifecycle_subparser(subparsers) -> None:
    task_p = subparsers.add_parser(
        "task",
        aliases=["task-lifecycle"],
        help="Unified command for end-to-end task and git lifecycle orchestration",
    )
    task_p.add_argument(
        "--phase",
        choices=["start", "finish", "scaffold", "sync"],
        default="start",
        help="Lifecycle phase to execute (start, finish, scaffold, sync)",
    )
    task_p.add_argument("--repo", default=None, help="Target GitHub repository (e.g. Chief-Strategist-J/quant-trading-platform)")
    task_p.add_argument("--package", default=None, help="Target package name (e.g. quant-trading-platform)")
    task_p.add_argument("--feature", default=None, help="Feature slug to scaffold or link")
    task_p.add_argument("--slug", default=None, help="Branch name slug")
    task_p.add_argument("--title", default=None, help="Task/Issue or PR title")
    task_p.add_argument("--body", default=None, help="Issue or PR description")
    task_p.add_argument("--issue", default=None, type=int, help="Issue number for finish phase")
    task_p.add_argument("--base-branch", default="main", help="Base branch (default: main)")
    task_p.add_argument("--test-cmd", default=None, help="Verification test command (e.g. 'cargo test', 'pytest')")
    task_p.add_argument("--no-sync", action="store_true", help="Skip parent submodule sync")
    task_p.add_argument("--submodule", default=None, help="Submodule name for sync phase")
    task_p.add_argument("--parent-path", default="..", help="Parent repository path for sync")
    task_p.add_argument("--base-dir", default=".", help="Base directory for scaffolding")
    task_p.add_argument("--json", action="store_true", help="Output result in JSON format")
    task_p.set_defaults(handler=handle_unified_task_command)

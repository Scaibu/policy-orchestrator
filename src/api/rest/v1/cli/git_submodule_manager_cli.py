"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER CLI IMPLEMENTATION
================================================================================
1. OVERVIEW & OBJECTIVE:
   CLI interface for managing multi-repo Git submodules:
     - add: registers submodule and stages 160000 gitlink commit.
     - update: updates single submodule pointer.
     - pull: batch updates all/specified submodules to latest upstream commits.
     - push: verifies child commits and updates parent pointer.
     - sync-status: audits commit drift.

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Posix exit codes: 0 for success, 1 for error.
================================================================================
"""

import argparse
import json
import os
import subprocess
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
)
from src.api.rest.v1.routers.git_submodule_manager_router import (
    get_git_submodule_manager_handler,
)


import os
import subprocess


def _resolve_default_repo() -> str:
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return out
    except Exception:
        pass
    return ""


def _resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GIT_TOKEN")
    if token:
        return token
    try:
        out = subprocess.check_output(["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            return out
    except Exception:
        pass
    try:
        import re
        out = subprocess.check_output(["gh", "auth", "status", "-t"], text=True, stderr=subprocess.STDOUT)
        match = re.search(r"Token:\s*(\S+)", out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None


def register_git_submodule_manager_subparser(subparsers: argparse._SubParsersAction) -> None:
    sub_parser = subparsers.add_parser("submodule", help="Multi-repo Git submodule management (add, update, pull, push, status)")
    sub_actions = sub_parser.add_subparsers(dest="action", required=True)

    add_p = sub_actions.add_parser("add", help="Add a new submodule to parent repository")
    add_p.add_argument("--repo", "-r", default=None, help="Target parent repository URL (auto-detected if omitted)")
    add_p.add_argument("--branch", "-b", default="main", help="Target branch name")
    add_p.add_argument("--path", "-p", required=True, help="Relative path for submodule in parent repo")
    add_p.add_argument("--sub-url", "-u", required=True, help="Child submodule remote repository URL")
    add_p.add_argument("--sub-ref", default="main", help="Child branch or commit SHA to link to")
    add_p.add_argument("--message", "-m", default="chore: add submodule", help="Commit message")
    add_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    add_p.add_argument("--author-name", default=None, help="Author name")
    add_p.add_argument("--author-email", default=None, help="Author email")
    add_p.add_argument("--token", "-t", default=None, help="Access token (auto-resolved from env/gh if omitted)")
    add_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    upd_p = sub_actions.add_parser("update", help="Update an existing submodule pointer")
    upd_p.add_argument("--repo", "-r", default=None, help="Target parent repository URL (auto-detected if omitted)")
    upd_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    upd_p.add_argument("--path", "-p", required=True, help="Relative path for submodule in parent repo")
    upd_p.add_argument("--target-sha", default=None, help="Explicit target commit SHA")
    upd_p.add_argument("--sub-branch", default=None, help="Submodule branch to pull latest HEAD from")
    upd_p.add_argument("--message", "-m", default="chore: update submodule pointer", help="Commit message")
    upd_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    upd_p.add_argument("--token", "-t", default=None, help="Access token (auto-resolved from env/gh if omitted)")
    upd_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    pull_p = sub_actions.add_parser("pull", help="Pull latest upstream commits for submodules")
    pull_p.add_argument("--repo", "-r", default=None, help="Target parent repository URL (auto-detected if omitted)")
    pull_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    pull_p.add_argument("--paths", nargs="*", default=None, help="Optional subset of submodule paths to pull")
    pull_p.add_argument("--message", "-m", default="chore: pull and sync submodules", help="Commit message")
    pull_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    pull_p.add_argument("--token", "-t", default=None, help="Access token (auto-resolved from env/gh if omitted)")
    pull_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    push_p = sub_actions.add_parser("push", help="Execute multi-repo push ensuring child commits exist on remotes")
    push_p.add_argument("--repo", "-r", default=None, help="Target parent repository URL (auto-detected if omitted)")
    push_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    push_p.add_argument("--paths", nargs="*", default=None, help="Optional list of submodule paths (all if omitted)")
    push_p.add_argument("--no-verify", action="store_false", dest="verify", help="Skip remote commit verification")
    push_p.add_argument("--message", "-m", default="chore: synchronize multi-repo submodules", help="Commit message")
    push_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    push_p.add_argument("--token", "-t", default=None, help="Access token (auto-resolved from env/gh if omitted)")
    push_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    stat_p = sub_actions.add_parser("status", help="Audit synchronization drift of all submodules")
    stat_p.add_argument("--repo", "-r", default=None, help="Target parent repository URL (auto-detected if omitted)")
    stat_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    stat_p.add_argument("--token", "-t", default=None, help="Access token (auto-resolved from env/gh if omitted)")
    stat_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    upg_p = sub_actions.add_parser("upgrade", help="Auto-update and self-heal the submodule manager code and dependencies")
    upg_p.add_argument("--branch", "-b", default="main", help="Branch to pull latest tool updates from")
    upg_p.add_argument("--json", action="store_true", help="Output JSON envelope")


def handle_git_submodule_manager_cli(args: argparse.Namespace) -> int:
    if args.action == "upgrade":
        import pathlib
        tool_dir = pathlib.Path(__file__).resolve().parents[5]
        res = subprocess.run(["git", "pull", "--rebase", "origin", args.branch], cwd=str(tool_dir), capture_output=True, text=True)
        is_success = res.returncode == 0
        details = {
            "action": "UPGRADE_TOOL",
            "tool_directory": str(tool_dir),
            "branch": args.branch,
            "success": is_success,
            "stdout": res.stdout.strip(),
            "stderr": res.stderr.strip() if not is_success else "",
        }
        print(json.dumps(details, indent=2))
        return 0 if is_success else 1
    handler = get_git_submodule_manager_handler()
    ctx = TraceContext(trace_id="cli-submod-trace", request_id="cli-submod-req", correlation_id="cli-submod-corr")

    repo_url = args.repo or _resolve_default_repo()
    if not repo_url:
        print("Error: Target repository not specified and could not be detected from git remote origin.")
        return 1

    token = _resolve_token(args.token)

    if args.action == "add":
        cmd = AddSubmoduleCommand(
            repo_url=repo_url,
            branch=args.branch,
            submodule_path=args.path,
            submodule_url=args.sub_url,
            submodule_ref=args.sub_ref,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            author_name=getattr(args, "author_name", None),
            author_email=getattr(args, "author_email", None),
            token=token,
        )
        res = handler.service.add_submodule(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    elif args.action == "update":
        cmd = UpdateSubmoduleCommand(
            repo_url=repo_url,
            branch=args.branch,
            submodule_path=args.path,
            target_commit_sha=args.target_sha,
            submodule_branch=args.sub_branch,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=token,
        )
        res = handler.service.update_submodule(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    elif args.action == "pull":
        cmd = PullSubmodulesCommand(
            repo_url=repo_url,
            branch=args.branch,
            submodule_paths=args.paths,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=token,
        )
        res = handler.service.pull_submodules(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    elif args.action == "push":
        cmd = PushMultiRepoCommand(
            parent_repo_url=repo_url,
            parent_branch=args.branch,
            submodule_paths=args.paths,
            verify_remote_exists=getattr(args, "verify", True),
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=token,
        )
        res = handler.service.push_multi_repo(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    elif args.action == "status":
        cmd = SubmoduleSyncStatusCommand(
            repo_url=repo_url,
            branch=args.branch,
            token=token,
        )
        res = handler.service.get_sync_status(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    return 0

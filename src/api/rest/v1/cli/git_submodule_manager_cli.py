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


def register_git_submodule_manager_subparser(subparsers: argparse._SubParsersAction) -> None:
    sub_parser = subparsers.add_parser("submodule", help="Multi-repo Git submodule management (add, update, pull, push, status)")
    sub_actions = sub_parser.add_subparsers(dest="action", required=True)

    add_p = sub_actions.add_parser("add", help="Add a new submodule to parent repository")
    add_p.add_argument("--repo", "-r", required=True, help="Target parent repository URL or owner/repo")
    add_p.add_argument("--branch", "-b", default="main", help="Target branch name")
    add_p.add_argument("--path", "-p", required=True, help="Relative path for submodule in parent repo")
    add_p.add_argument("--sub-url", "-u", required=True, help="Child submodule remote repository URL")
    add_p.add_argument("--sub-ref", default="main", help="Child branch or commit SHA to link to")
    add_p.add_argument("--message", "-m", default="chore: add submodule", help="Commit message")
    add_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    add_p.add_argument("--author-name", default=None, help="Author name")
    add_p.add_argument("--author-email", default=None, help="Author email")
    add_p.add_argument("--token", "-t", default=None, help="Access token")
    add_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    upd_p = sub_actions.add_parser("update", help="Update an existing submodule pointer")
    upd_p.add_argument("--repo", "-r", required=True, help="Target parent repository URL")
    upd_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    upd_p.add_argument("--path", "-p", required=True, help="Relative path for submodule in parent repo")
    upd_p.add_argument("--target-sha", default=None, help="Explicit target commit SHA")
    upd_p.add_argument("--sub-branch", default=None, help="Submodule branch to pull latest HEAD from")
    upd_p.add_argument("--message", "-m", default="chore: update submodule pointer", help="Commit message")
    upd_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    upd_p.add_argument("--token", "-t", default=None, help="Access token")
    upd_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    pull_p = sub_actions.add_parser("pull", help="Pull latest upstream commits for submodules")
    pull_p.add_argument("--repo", "-r", required=True, help="Target parent repository URL")
    pull_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    pull_p.add_argument("--paths", nargs="*", default=None, help="Optional subset of submodule paths to pull")
    pull_p.add_argument("--message", "-m", default="chore: pull and sync submodules", help="Commit message")
    pull_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    pull_p.add_argument("--token", "-t", default=None, help="Access token")
    pull_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    push_p = sub_actions.add_parser("push", help="Execute multi-repo push ensuring child commits exist on remotes")
    push_p.add_argument("--repo", "-r", required=True, help="Target parent repository URL")
    push_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    push_p.add_argument("--paths", nargs="+", required=True, help="List of submodule paths to verify before pushing")
    push_p.add_argument("--no-verify", action="store_false", dest="verify", help="Skip remote commit verification")
    push_p.add_argument("--message", "-m", default="chore: synchronize multi-repo submodules", help="Commit message")
    push_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    push_p.add_argument("--token", "-t", default=None, help="Access token")
    push_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    stat_p = sub_actions.add_parser("status", help="Audit synchronization drift of all submodules")
    stat_p.add_argument("--repo", "-r", required=True, help="Target parent repository URL")
    stat_p.add_argument("--branch", "-b", default="main", help="Target parent branch name")
    stat_p.add_argument("--token", "-t", default=None, help="Access token")
    stat_p.add_argument("--json", action="store_true", help="Output JSON envelope")


def handle_git_submodule_manager_cli(args: argparse.Namespace) -> int:
    handler = get_git_submodule_manager_handler()
    ctx = TraceContext(trace_id="cli-submod-trace", request_id="cli-submod-req", correlation_id="cli-submod-corr")

    if args.action == "add":
        cmd = AddSubmoduleCommand(
            repo_url=args.repo,
            branch=args.branch,
            submodule_path=args.path,
            submodule_url=args.sub_url,
            submodule_ref=args.sub_ref,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            author_name=getattr(args, "author_name", None),
            author_email=getattr(args, "author_email", None),
            token=args.token,
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
            repo_url=args.repo,
            branch=args.branch,
            submodule_path=args.path,
            target_commit_sha=args.target_sha,
            submodule_branch=args.sub_branch,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=args.token,
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
            repo_url=args.repo,
            branch=args.branch,
            submodule_paths=args.paths,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=args.token,
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
            parent_repo_url=args.repo,
            parent_branch=args.branch,
            submodule_paths=args.paths,
            verify_remote_exists=getattr(args, "verify", True),
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=args.token,
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
            repo_url=args.repo,
            branch=args.branch,
            token=args.token,
        )
        res = handler.service.get_sync_status(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    return 0

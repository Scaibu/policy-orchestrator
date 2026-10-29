import argparse
import json
from src.shared.types.envelope_types import TraceContext
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    DeleteFileCommand,
)
from src.api.rest.v1.routers.git_remote_mutator_router import get_git_remote_mutator_handler


def register_git_remote_mutator_subparser(subparsers: argparse._SubParsersAction) -> None:
    mut_parser = subparsers.add_parser("remote-mutate", help="Zero-clone remote Git mutation (single-file, batch, delete)")
    mut_sub = mut_parser.add_subparsers(dest="action", required=True)

    file_p = mut_sub.add_parser("file", help="Mutate a single remote file")
    file_p.add_argument("--repo", "-r", required=True, help="Target repository URL or owner/repo")
    file_p.add_argument("--branch", "-b", default="main", help="Target branch name")
    file_p.add_argument("--path", "-p", required=True, help="Relative file path")
    file_p.add_argument("--content", "-c", required=True, help="File content string")
    file_p.add_argument("--message", "-m", required=True, help="Commit message")
    file_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    file_p.add_argument("--author-name", default=None, help="Author name")
    file_p.add_argument("--author-email", default=None, help="Author email")
    file_p.add_argument("--token", "-t", default=None, help="Access token")
    file_p.add_argument("--json", action="store_true", help="Output JSON envelope")

    del_p = mut_sub.add_parser("delete", help="Delete a remote file")
    del_p.add_argument("--repo", "-r", required=True, help="Target repository URL or owner/repo")
    del_p.add_argument("--branch", "-b", default="main", help="Target branch name")
    del_p.add_argument("--path", "-p", required=True, help="Relative file path")
    del_p.add_argument("--message", "-m", required=True, help="Commit message")
    del_p.add_argument("--expected-sha", default=None, help="Expected parent SHA for OCC")
    del_p.add_argument("--token", "-t", default=None, help="Access token")
    del_p.add_argument("--json", action="store_true", help="Output JSON envelope")


def handle_git_remote_mutator_cli(args: argparse.Namespace) -> int:
    handler = get_git_remote_mutator_handler()
    ctx = TraceContext(trace_id="cli-mut-trace", request_id="cli-mut-req", correlation_id="cli-mut-corr")

    if args.action == "file":
        cmd = MutateFileCommand(
            repo_url=args.repo,
            branch=args.branch,
            file_path=args.path,
            content=args.content,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            author_name=getattr(args, "author_name", None),
            author_email=getattr(args, "author_email", None),
            token=args.token,
        )
        res = handler.service.mutate_file(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    elif args.action == "delete":
        cmd = DeleteFileCommand(
            repo_url=args.repo,
            branch=args.branch,
            file_path=args.path,
            commit_message=args.message,
            expected_parent_sha=args.expected_sha,
            token=args.token,
        )
        res = handler.service.delete_file(ctx, cmd)
        if res.is_ok:
            data = res.unwrap().model_dump()
            print(json.dumps(data, indent=2))
            return 0
        print(f"Error: {res.error()}")
        return 1

    return 0

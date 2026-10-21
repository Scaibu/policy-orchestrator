"""
================================================================================
BLUEPRINT: GIT VERIFIER CLI ADAPTER
================================================================================
1. OVERVIEW:
   CLI delivery interface for Git Merkle tree state audits and submodule
   gitlink verification (mode 160000).
2. SUBCOMMANDS / OPTIONS:
   - --repo <dir>: Path to git repository (default: current directory).
   - --no-submodules: Skip submodule commit verification.
   - --json: Output result in JSON format.
3. ZERO-INLINE-COMMENT DOCTRINE:
   All arguments, parsing logic, and execution steps are documented in this
   top-level blueprint. Method bodies remain 100% comment-free.
================================================================================
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[5])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features.git_verifier.contracts.git_verifier_types import VerifyGitCommand
from src.features.git_verifier.contracts.git_verifier_schema import GitVerifierSchema
from src.features.git_verifier.service.git_verifier_service import GitVerifierDomainService
from src.features.git_verifier.infrastructure.adapters.git_verifier_repository_adapter import (
    AlloyDBGitVerifierRepository,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="git-verifier-cli",
        description="Git Merkle Tree & Submodule (Mode 160000) Verification CLI",
    )
    parser.add_argument("--repo", "-r", default=".", help="Root path of git repository (default: current directory)")
    parser.add_argument("--no-submodules", action="store_true", help="Skip checking submodule pointers")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser


def handle_git_verifier_cli(args: argparse.Namespace) -> int:
    try:
        db_url = AppConfigLoader.get_database_url()
        repo = AlloyDBGitVerifierRepository(db_url=db_url)
        service = GitVerifierDomainService(repository=repo)

        cmd = VerifyGitCommand(
            repo_path=str(Path(args.repo).resolve()),
            check_submodules=not args.no_submodules,
        )

        ctx = TraceContext(
            trace_id="cli-git-verifier",
            request_id="cli-req-git-verifier",
            correlation_id="cli-corr-git-verifier",
        )

        result = service.verify_git(ctx, cmd)
        if result.is_err:
            print(f"Error executing git verification: {result.error()}", file=sys.stderr)
            return 1

        res = result.unwrap()
        if args.json:
            print(json.dumps(GitVerifierSchema.to_api_verify_git(res), indent=2))
        else:
            print("=" * 80)
            print("GIT REPOSITORY VERIFICATION REPORT")
            print("=" * 80)
            print(f"Repository Path:   {res.repo_path}")
            print(f"Current Branch:    {res.current_branch or 'N/A'}")
            print(f"HEAD Commit:       {res.head_commit or 'N/A'}")
            print(f"Is Detached:       {'YES' if res.is_detached else 'NO'}")
            if res.remote.upstream_branch:
                sync_label = "IN SYNC" if res.remote.is_in_sync else f"AHEAD {res.remote.ahead_count}, BEHIND {res.remote.behind_count}"
                print(f"Upstream Remote:   {res.remote.upstream_branch} [{sync_label}]")
            print(f"Working Tree:      {'CLEAN' if res.working_tree.is_clean else 'DIRTY'}")
            if res.working_tree.staged_files:
                print(f"  Staged ({len(res.working_tree.staged_files)}):")
                for f in res.working_tree.staged_files[:10]:
                    print(f"    + {f}")
            if res.working_tree.unstaged_files:
                print(f"  Unstaged ({len(res.working_tree.unstaged_files)}):")
                for f in res.working_tree.unstaged_files[:10]:
                    print(f"    * {f}")
            if res.working_tree.untracked_files:
                print(f"  Untracked ({len(res.working_tree.untracked_files)}):")
                for f in res.working_tree.untracked_files[:10]:
                    print(f"    ? {f}")
            print(f"Submodules Found:  {len(res.submodules)}")
            print(f"Duration:          {res.duration_ms} ms")
            print(f"Status:            {res.status}")
            if res.submodules:
                print("-" * 80)
                print("Submodule Pointer Status:")
                for sub in res.submodules:
                    sync_icon = "SYNCED" if sub.is_synchronized else "MISMATCH"
                    dirty_tag = " (DIRTY)" if sub.is_dirty else ""
                    print(f"  [{sync_icon}] {sub.name} ({sub.path}){dirty_tag} -> {sub.actual_commit or 'MISSING'}")
            print("=" * 80)
        return 0 if res.is_clean else 1
    except Exception as exc:
        print(f"Git verification failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(handle_git_verifier_cli(args))


if __name__ == "__main__":
    main()

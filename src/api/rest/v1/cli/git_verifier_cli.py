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
            print(f"Is Detached:       {'YES' if getattr(res, 'is_detached', False) else 'NO'}")
            remote = getattr(res, "remote", None)
            if remote and getattr(remote, "upstream_branch", None):
                sync_label = "IN SYNC" if getattr(remote, "is_in_sync", True) else f"AHEAD {getattr(remote, 'ahead_count', 0)}, BEHIND {getattr(remote, 'behind_count', 0)}"
                print(f"Upstream Remote:   {remote.upstream_branch} [{sync_label}]")
            wt = getattr(res, "working_tree", None)
            if wt is not None:
                print(f"Working Tree:      {'CLEAN' if getattr(wt, 'is_clean', True) else 'DIRTY'}")
                if getattr(wt, "staged_files", None):
                    print(f"  Staged ({len(wt.staged_files)}):")
                    for f in wt.staged_files[:10]:
                        print(f"    + {f}")
                if getattr(wt, "unstaged_files", None):
                    print(f"  Unstaged ({len(wt.unstaged_files)}):")
                    for f in wt.unstaged_files[:10]:
                        print(f"    * {f}")
                if getattr(wt, "untracked_files", None):
                    print(f"  Untracked ({len(wt.untracked_files)}):")
                    for f in wt.untracked_files[:10]:
                        print(f"    ? {f}")
            else:
                print(f"Is Clean:          {'YES' if getattr(res, 'is_clean', True) else 'DRIFT DETECTED'}")
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

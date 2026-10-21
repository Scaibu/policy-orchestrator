"""
================================================================================
BLUEPRINT: FILE SEARCH CLI ADAPTER
================================================================================
1. OVERVIEW:
   CLI delivery interface for recursive file and directory tree search using
   DFS / BFS algorithms and inode caching.
2. SUBCOMMANDS / OPTIONS:
   - --root <dir>: Directory path to start traversal (required).
   - --pattern <pat>: Filename pattern or extension filter (optional).
   - --algorithm {DFS,BFS}: Traversal algorithm selection (default: DFS).
   - --max-depth <int>: Maximum recursion depth (default: 10).
   - --exclude <dirs...>: Space-separated directory names to prune.
   - --json: Output result in JSON envelope.
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

from src.features.file_search.contracts.file_search_types import FileSearchCommand
from src.features.file_search.contracts.file_search_schema import FileSearchSchema
from src.features.file_search.service.file_search_service import FileSearchDomainService
from src.features.file_search.infrastructure.adapters.file_search_repository_adapter import (
    AlloyDBFileSearchRepository,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="file-search-cli",
        description="High-Throughput File Tree Traversal CLI (DFS / BFS)",
    )
    parser.add_argument("--root", "--dir", dest="root", required=True, help="Root directory path to traverse")
    parser.add_argument("--pattern", default=None, help="Pattern or extension filter (e.g. .py, *test*)")
    parser.add_argument("--algorithm", choices=["DFS", "BFS"], default="DFS", help="Traversal algorithm (DFS/BFS)")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum traversal depth (default: 10)")
    parser.add_argument("--exclude", nargs="*", default=None, help="Additional directories to prune")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser


def handle_file_search_cli(args: argparse.Namespace) -> int:
    try:
        db_url = AppConfigLoader.get_database_url()
        repo = AlloyDBFileSearchRepository(db_url=db_url)
        service = FileSearchDomainService(repository=repo)

        exclude_dirs = args.exclude if args.exclude is not None else [
            ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"
        ]

        cmd = FileSearchCommand(
            root_path=args.root,
            pattern=args.pattern,
            algorithm=args.algorithm,
            max_depth=args.max_depth,
            exclude_dirs=exclude_dirs,
        )

        ctx = TraceContext(
            trace_id="cli-file-search",
            request_id="cli-req-file-search",
            correlation_id="cli-corr-file-search",
        )

        result = service.search_files(ctx, cmd)
        if result.is_err:
            print(f"Error executing file search: {result.error()}", file=sys.stderr)
            return 1

        res = result.unwrap()
        if args.json:
            print(json.dumps(FileSearchSchema.to_api_search_files(res), indent=2))
        else:
            print("=" * 80)
            print("FILE SEARCH TRAVERSAL REPORT")
            print("=" * 80)
            print(f"Root Directory:    {res.root_path}")
            print(f"Algorithm Used:    {cmd.algorithm}")
            print(f"Total Found:       {res.total_found}")
            print(f"Duration:          {res.duration_ms} ms")
            print(f"Status:            {res.status}")
            print("-" * 80)
            for item in res.files[:50]:
                print(f"  [{item.depth}] {item.path} ({item.size_bytes} bytes)")
            if len(res.files) > 50:
                print(f"  ... and {len(res.files) - 50} more files")
            print("=" * 80)
        return 0
    except Exception as exc:
        print(f"File search failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(handle_file_search_cli(args))


if __name__ == "__main__":
    main()

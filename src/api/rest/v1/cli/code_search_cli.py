"""
================================================================================
BLUEPRINT: CODE SEARCH CLI ADAPTER
================================================================================
1. OVERVIEW:
   CLI delivery interface for pattern matching across code files using
   Aho-Corasick DFA or Boyer-Moore-Horspool algorithms.
2. SUBCOMMANDS / OPTIONS:
   - --patterns <p1> <p2>...: Patterns to search (required).
   - --files <f1> <f2>...: File paths to scan (required).
   - --algorithm {AHO_CORASICK,BOYER_MOORE}: Matching algorithm (default: AHO_CORASICK).
   - --case-insensitive: Perform case-insensitive search.
   - --max-matches <int>: Max matches per file (default: 100).
   - --json: Output results in JSON format.
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

from src.features.code_search.contracts.code_search_types import CodeSearchCommand
from src.features.code_search.contracts.code_search_schema import CodeSearchSchema
from src.features.code_search.service.code_search_service import CodeSearchDomainService
from src.features.code_search.infrastructure.adapters.code_search_repository_adapter import (
    AlloyDBCodeSearchRepository,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-search-cli",
        description="Multi-Pattern Code Search CLI (Aho-Corasick & Boyer-Moore-Horspool)",
    )
    parser.add_argument("--patterns", "-p", nargs="+", required=True, help="Patterns to search for")
    parser.add_argument("--files", "-f", nargs="+", required=True, help="Target files to scan")
    parser.add_argument(
        "--algorithm",
        choices=["AHO_CORASICK", "BOYER_MOORE"],
        default="AHO_CORASICK",
        help="Algorithm: AHO_CORASICK or BOYER_MOORE",
    )
    parser.add_argument("--case-insensitive", "-i", action="store_true", help="Case-insensitive search")
    parser.add_argument("--max-matches", type=int, default=100, help="Max matches per file")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser


def handle_code_search_cli(args: argparse.Namespace) -> int:
    try:
        db_url = AppConfigLoader.get_database_url()
        repo = AlloyDBCodeSearchRepository(db_url=db_url)
        service = CodeSearchDomainService(repository=repo)

        cmd = CodeSearchCommand(
            patterns=args.patterns,
            target_files=args.files,
            algorithm=args.algorithm,
            case_sensitive=not args.case_insensitive,
            max_matches_per_file=args.max_matches,
        )

        ctx = TraceContext(
            trace_id="cli-code-search",
            request_id="cli-req-code-search",
            correlation_id="cli-corr-code-search",
        )

        result = service.search_code(ctx, cmd)
        if result.is_err:
            print(f"Error executing code search: {result.error()}", file=sys.stderr)
            return 1

        res = result.unwrap()
        if args.json:
            print(json.dumps(CodeSearchSchema.to_api_search_code(res), indent=2))
        else:
            print("=" * 80)
            print("CODE SEARCH MATCH REPORT")
            print("=" * 80)
            print(f"Algorithm:         {cmd.algorithm}")
            print(f"Patterns:          {', '.join(cmd.patterns)}")
            print(f"Files Scanned:     {res.scanned_files_count}")
            print(f"Total Matches:     {res.total_matches}")
            print(f"Duration:          {res.duration_ms} ms")
            print(f"Status:            {res.status}")
            print("-" * 80)
            for m in res.matches[:50]:
                print(f"  {m.file_path}:{m.line_number} [{m.pattern}] -> {m.line_content}")
            if len(res.matches) > 50:
                print(f"  ... and {len(res.matches) - 50} more matches")
            print("=" * 80)
        return 0
    except Exception as exc:
        print(f"Code search failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(handle_code_search_cli(args))


if __name__ == "__main__":
    main()

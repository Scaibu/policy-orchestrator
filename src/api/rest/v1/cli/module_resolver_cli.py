"""
================================================================================
BLUEPRINT: MODULE RESOLVER CLI ADAPTER
================================================================================
1. OVERVIEW:
   CLI delivery interface for Python module resolution per PEP 302 / PEP 451.
   Inspects sequential sys.path entries, detects collisions & shadowing.
2. SUBCOMMANDS / OPTIONS:
   - --module <name>: Dotted module path to resolve (required).
   - --search-paths <paths...>: Directories simulating sys.path (default: current workspace root).
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

from src.features.module_resolver.contracts.module_resolver_types import ResolveModuleCommand
from src.features.module_resolver.contracts.module_resolver_schema import ModuleResolverSchema
from src.features.module_resolver.service.module_resolver_service import ModuleResolverDomainService
from src.features.module_resolver.infrastructure.adapters.module_resolver_repository_adapter import (
    AlloyDBModuleResolverRepository,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="module-resolver-cli",
        description="Python Module Resolution & Collision Audit CLI (PEP 302/451)",
    )
    parser.add_argument("--module", "-m", required=True, help="Dotted module path (e.g. src.features.file_search)")
    parser.add_argument(
        "--search-paths",
        "-s",
        nargs="+",
        default=[REPO_ROOT],
        help="Directories simulating sys.path (default: workspace root)",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser


def handle_module_resolver_cli(args: argparse.Namespace) -> int:
    try:
        db_url = AppConfigLoader.get_database_url()
        repo = AlloyDBModuleResolverRepository(db_url=db_url)
        service = ModuleResolverDomainService(repository=repo)

        cmd = ResolveModuleCommand(
            module_name=args.module,
            search_paths=args.search_paths,
        )

        ctx = TraceContext(
            trace_id="cli-module-resolver",
            request_id="cli-req-module-resolver",
            correlation_id="cli-corr-module-resolver",
        )

        result = service.resolve_module(ctx, cmd)
        if result.is_err:
            print(f"Error resolving module: {result.error()}", file=sys.stderr)
            return 1

        res = result.unwrap()
        if args.json:
            print(json.dumps(ModuleResolverSchema.to_api_resolve_module(res), indent=2))
        else:
            print("=" * 80)
            print("MODULE RESOLUTION REPORT (PEP 302/451/566)")
            print("=" * 80)
            print(f"Module Name:       {res.module_name}")
            print(f"Is Valid:          {'YES' if res.is_valid else 'NO'}")
            print(f"Package Type:      {res.package_type}")
            print(f"Resolved Path:     {res.resolved_path or 'BUILTIN / NOT FOUND'}")
            print(f"Distribution:      {res.distribution_name or 'N/A'}")
            print(f"Installed Version: {res.version or 'N/A'}")
            print(f"Is Builtin:        {'YES' if res.is_builtin else 'NO'}")
            print(f"Is Standard Lib:   {'YES' if res.is_stdlib else 'NO'}")
            print(f"Has Collision:     {'YES (SHADOWED)' if res.has_collision else 'NO'}")
            if res.shadowed_paths:
                print("Shadowed Paths:")
                for p in res.shadowed_paths:
                    print(f"  - {p}")
            print(f"Duration:          {res.duration_ms} ms")
            print(f"Status:            {res.status}")
            print("=" * 80)
        return 0 if res.is_valid else 1
    except Exception as exc:
        print(f"Module resolution failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(handle_module_resolver_cli(args))


if __name__ == "__main__":
    main()

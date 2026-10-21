"""
================================================================================
BLUEPRINT: FILE SEARCH & CODEBASE INTELLIGENCE CLI ADAPTER
================================================================================
1. OVERVIEW:
   Unified CLI interface for file search and codebase inspection.
   Supports:
   - Full recursive file tree construction (ASCII tree visual or JSON).
   - Multi-pattern code search (Aho-Corasick DFA & Boyer-Moore-Horspool).
   - Python module resolution (PEP 302/451) and collision/shadowing detection.
   - Git repository Merkle tree & submodule (mode 160000) verification.
   All functionality can be invoked in a single unified execution or selectively.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All arguments, parsing logic, and execution steps are documented in this
   top-level blueprint. Method bodies remain 100% comment-free.
================================================================================
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

REPO_ROOT = str(Path(__file__).resolve().parents[5])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features.file_search.contracts.file_search_types import FileSearchCommand, TreeNode
from src.features.file_search.contracts.file_search_schema import FileSearchSchema
from src.features.file_search.service.file_search_service import FileSearchDomainService
from src.features.file_search.infrastructure.adapters.file_search_repository_adapter import (
    AlloyDBFileSearchRepository,
)
from src.features.code_search.contracts.code_search_types import CodeSearchCommand
from src.features.code_search.service.code_search_service import CodeSearchDomainService
from src.features.code_search.infrastructure.adapters.code_search_repository_adapter import (
    AlloyDBCodeSearchRepository,
)
from src.features.module_resolver.contracts.module_resolver_types import ResolveModuleCommand
from src.features.module_resolver.service.module_resolver_service import ModuleResolverDomainService
from src.features.module_resolver.infrastructure.adapters.module_resolver_repository_adapter import (
    AlloyDBModuleResolverRepository,
)
from src.features.git_verifier.contracts.git_verifier_types import VerifyGitCommand
from src.features.git_verifier.service.git_verifier_service import GitVerifierDomainService
from src.features.git_verifier.infrastructure.adapters.git_verifier_repository_adapter import (
    AlloyDBGitVerifierRepository,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="file-search-cli",
        description="Unified Codebase File Tree, Code Search, Module Resolution & Git Verification CLI",
    )
    parser.add_argument("--root", "--dir", dest="root", default=".", help="Root directory path to traverse (default: .)")
    parser.add_argument("--pattern", default=None, help="Filename pattern or extension filter (e.g. .py, *test*)")
    parser.add_argument("--algorithm", choices=["DFS", "BFS"], default="DFS", help="Traversal algorithm (DFS/BFS)")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum traversal depth (default: 10)")
    parser.add_argument("--exclude", nargs="*", default=None, help="Additional directories to prune")
    parser.add_argument("--tree", action="store_true", default=True, help="Print complete visual hierarchical file tree")
    parser.add_argument("--code-search", "-p", nargs="*", default=None, help="Optional patterns to search in found files")
    parser.add_argument("--resolve-module", "-m", nargs="*", default=None, help="Optional Python modules to resolve")
    parser.add_argument("--verify-git", action="store_true", help="Verify Git state and submodule pointers")
    parser.add_argument("--all", action="store_true", help="Execute complete scan: file tree, git verification, and module checks")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser


def render_ascii_tree(node: TreeNode, prefix: str = "", is_last: bool = True) -> List[str]:
    lines = []
    connector = "└── " if is_last else "├── "
    icon = "📁 " if node.is_dir else "📄 "
    size_str = f" ({node.size_bytes} B)" if not node.is_dir else ""
    lines.append(f"{prefix}{connector}{icon}{node.name}{size_str}")

    child_prefix = prefix + ("    " if is_last else "│   ")
    for idx, child in enumerate(node.children):
        last_child = (idx == len(node.children) - 1)
        lines.extend(render_ascii_tree(child, child_prefix, last_child))

    return lines


def handle_file_search_cli(args: argparse.Namespace) -> int:
    try:
        db_url = AppConfigLoader.get_database_url()
        root_path = str(Path(args.root).resolve())

        ctx = TraceContext(
            trace_id="cli-file-search-all",
            request_id="cli-req-file-search-all",
            correlation_id="cli-corr-file-search-all",
        )

        exclude_dirs = args.exclude if args.exclude is not None else [
            ".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", ".vscode"
        ]

        file_repo = AlloyDBFileSearchRepository(db_url=db_url)
        file_service = FileSearchDomainService(repository=file_repo)
        cmd = FileSearchCommand(
            root_path=root_path,
            pattern=args.pattern,
            algorithm=args.algorithm,
            max_depth=args.max_depth,
            exclude_dirs=exclude_dirs,
        )

        file_res = file_service.search_files(ctx, cmd)
        if file_res.is_err:
            print(f"Error executing file search: {file_res.error()}", file=sys.stderr)
            return 1
        res = file_res.unwrap()

        code_res = None
        patterns = args.code_search
        if patterns and res.files:
            code_repo = AlloyDBCodeSearchRepository(db_url=db_url)
            code_service = CodeSearchDomainService(repository=code_repo)
            target_files = [f.path for f in res.files if f.path.endswith((".py", ".yml", ".yaml", ".json", ".toml", ".md", ".sh"))]
            code_cmd = CodeSearchCommand(
                patterns=patterns,
                target_files=target_files[:200],
                algorithm="AHO_CORASICK",
            )
            code_res = code_service.search_code(ctx, code_cmd).unwrap()

        module_results = []
        modules_to_check = args.resolve_module
        if modules_to_check:
            mod_repo = AlloyDBModuleResolverRepository(db_url=db_url)
            mod_service = ModuleResolverDomainService(repository=mod_repo)
            for mod in modules_to_check:
                mod_cmd = ResolveModuleCommand(module_name=mod, search_paths=[root_path])
                module_results.append(mod_service.resolve_module(ctx, mod_cmd).unwrap())

        git_val = None
        if args.verify_git or args.all:
            git_repo = AlloyDBGitVerifierRepository(db_url=db_url)
            git_service = GitVerifierDomainService(repository=git_repo)
            git_cmd = VerifyGitCommand(repo_path=root_path, check_submodules=True)
            git_res = git_service.verify_git(ctx, git_cmd)
            if git_res.is_ok:
                git_val = git_res.unwrap()

        if args.json:
            payload: Dict[str, Any] = {
                "file_search": FileSearchSchema.to_api_search_files(res),
            }
            if code_res:
                payload["code_search"] = code_res.model_dump()
            if module_results:
                payload["module_resolver"] = [m.model_dump() for m in module_results]
            if git_val:
                payload["git_verifier"] = git_val.model_dump()
            print(json.dumps(payload, indent=2))
        else:
            print("=" * 80)
            print("COMPLETE CODEBASE & FILE TREE REPORT")
            print("=" * 80)
            print(f"Root Directory:    {res.root_path}")
            print(f"Total Directories: {res.total_dirs}")
            print(f"Total Files:       {res.total_found}")
            print(f"Traversal Time:    {res.duration_ms} ms")
            print(f"Status:            {res.status}")

            if git_val:
                print("-" * 80)
                print("GIT STATE & SUBMODULES:")
                print(f"  Branch:          {git_val.current_branch}")
                print(f"  HEAD Commit:     {git_val.head_commit or 'N/A'}")
                wt = getattr(git_val, "working_tree", None)
                if wt is not None:
                    print(f"  Working Tree:    {'CLEAN' if wt.is_clean else 'DIRTY'}")
                else:
                    print(f"  Is Clean:        {'YES' if git_val.is_clean else 'DRIFT DETECTED'}")
                remote = getattr(git_val, "remote", None)
                if remote and remote.upstream_branch:
                    sync_tag = "IN SYNC" if remote.is_in_sync else f"AHEAD {remote.ahead_count}, BEHIND {remote.behind_count}"
                    print(f"  Upstream Remote: {remote.upstream_branch} [{sync_tag}]")
                print(f"  Submodules:      {len(git_val.submodules)}")

            if res.tree:
                print("-" * 80)
                print("COMPLETE HIERARCHICAL FILE TREE:")
                print("-" * 80)
                for line in render_ascii_tree(res.tree, prefix="", is_last=True):
                    print(line)

            if code_res:
                print("-" * 80)
                print(f"CODE SEARCH MATCHES ({code_res.total_matches} found):")
                for m in code_res.matches[:25]:
                    print(f"  {m.file_path}:{m.line_number} [{m.pattern}] -> {m.line_content}")
                if len(code_res.matches) > 25:
                    print(f"  ... and {len(code_res.matches) - 25} more matches")

            if module_results:
                print("-" * 80)
                print("MODULE RESOLUTIONS (PEP 302/451):")
                for m in module_results:
                    status_str = "VALID" if m.is_valid else "INVALID"
                    print(f"  [{status_str}] {m.module_name} -> {m.resolved_path or 'NOT FOUND'} ({m.package_type})")

            print("=" * 80)
        return 0
    except Exception as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(handle_file_search_cli(args))


if __name__ == "__main__":
    main()

"""
================================================================================
BLUEPRINT: GIT REMOTE INSPECTOR CLI ADAPTER
================================================================================
1. OVERVIEW:
   CLI delivery interface for Zero-Clone Git Remote Inspection:
   - discover-contracts: Extract API specifications, data contracts, and routers.
   - branches: List remote branches and detect default branch.
   - tree: Output visual or JSON folder tree hierarchy.
   - search: Search files by pattern in remote repository.
   - content: Fetch decoded file content from remote repository.
   - diff: Compare two refs remotely without cloning.
2. ZERO-INLINE-COMMENT DOCTRINE:
   Method bodies remain 100% comment-free.
================================================================================
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[5])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    GetTreeHierarchyCommand,
    ListBranchesCommand,
    DiscoverContractsCommand,
    TraceCallChainCommand,
    IntegrateContextCommand,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_deps import (
    GitRemoteInspectorServiceDeps,
)
from src.features.git_remote_inspector.service.git_remote_inspector_service import (
    GitRemoteInspectorDomainService,
)
from src.features.git_remote_inspector.infrastructure.adapters.github_remote_client_adapter import (
    GitHubRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.gitlab_remote_client_adapter import (
    GitLabRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.bitbucket_remote_client_adapter import (
    BitbucketRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.composite_git_remote_client_adapter import (
    CompositeGitRemoteClientAdapter,
)
from src.features.git_remote_inspector.infrastructure.adapters.alloydb_git_remote_inspector_repository_adapter import (
    AlloyDBGitRemoteInspectorRepositoryAdapter,
)
from src.infra.config.config_loader import AppConfigLoader
from src.shared.types.envelope_types import TraceContext


def _create_service() -> GitRemoteInspectorDomainService:
    db_url = AppConfigLoader.get_database_url()
    gh_cfg = AppConfigLoader.get_github_config()
    gl_cfg = AppConfigLoader.get_gitlab_config()
    bb_cfg = AppConfigLoader.get_bitbucket_config()
    repo = AlloyDBGitRemoteInspectorRepositoryAdapter(db_url=db_url)
    gh_client = GitHubRemoteClientAdapter(
        api_base_url=gh_cfg.get("api_base_url", "https://api.github.com"),
        default_token=gh_cfg.get("token", ""),
        timeout_seconds=int(gh_cfg.get("timeout_seconds", 30)),
    )
    gl_client = GitLabRemoteClientAdapter(
        api_base_url=gl_cfg.get("api_base_url", "https://gitlab.com"),
        default_token=gl_cfg.get("token", ""),
        timeout_seconds=int(gl_cfg.get("timeout_seconds", 30)),
    )
    bb_client = BitbucketRemoteClientAdapter(
        api_base_url=bb_cfg.get("api_base_url", "https://api.bitbucket.org/2.0"),
        default_token=bb_cfg.get("token", ""),
        timeout_seconds=int(bb_cfg.get("timeout_seconds", 30)),
    )
    composite_client = CompositeGitRemoteClientAdapter(
        github_client=gh_client,
        gitlab_client=gl_client,
        bitbucket_client=bb_client,
    )
    deps = GitRemoteInspectorServiceDeps(
        repository=repo,
        remote_client=composite_client,
        config=gh_cfg,
    )
    return GitRemoteInspectorDomainService(deps=deps)


def register_git_remote_inspector_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("remote-inspect", help="Zero-clone remote Git repository inspection and contract discovery")
    inspect_subs = p.add_subparsers(dest="inspect_action", required=True)

    discover_p = inspect_subs.add_parser("discover-contracts", help="Discover and categorize API & data contracts")
    discover_p.add_argument("--repo", "-r", required=True, help="Target repository (e.g. owner/repo or URL)")
    discover_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    discover_p.add_argument("--token", default=None, help="Personal access token")
    discover_p.add_argument("--json", action="store_true", help="Output in JSON format")

    branches_p = inspect_subs.add_parser("branches", help="List remote repository branches")
    branches_p.add_argument("--repo", "-r", required=True, help="Target repository")
    branches_p.add_argument("--token", default=None, help="Personal access token")
    branches_p.add_argument("--json", action="store_true", help="Output in JSON format")

    tree_p = inspect_subs.add_parser("tree", help="Fetch remote folder and file tree hierarchy")
    tree_p.add_argument("--repo", "-r", required=True, help="Target repository")
    tree_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    tree_p.add_argument("--max-depth", type=int, default=None, help="Maximum traversal depth")
    tree_p.add_argument("--token", default=None, help="Personal access token")
    tree_p.add_argument("--json", action="store_true", help="Output in JSON format")

    search_p = inspect_subs.add_parser("search", help="Search remote files by pattern")
    search_p.add_argument("--repo", "-r", required=True, help="Target repository")
    search_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    search_p.add_argument("--pattern", "-p", default=None, help="Glob pattern (e.g. *schema*)")
    search_p.add_argument("--token", default=None, help="Personal access token")
    search_p.add_argument("--json", action="store_true", help="Output in JSON format")

    content_p = inspect_subs.add_parser("content", help="Read remote file content")
    content_p.add_argument("--repo", "-r", required=True, help="Target repository")
    content_p.add_argument("--path", "-f", required=True, help="Remote file path")
    content_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    content_p.add_argument("--token", default=None, help="Personal access token")
    content_p.add_argument("--json", action="store_true", help="Output in JSON format")

    diff_p = inspect_subs.add_parser("diff", help="Compare two remote branches/commits")
    diff_p.add_argument("--repo", "-r", required=True, help="Target repository")
    diff_p.add_argument("--base", required=True, help="Base ref or commit SHA")
    diff_p.add_argument("--head", required=True, help="Head ref or commit SHA")
    diff_p.add_argument("--token", default=None, help="Personal access token")
    diff_p.add_argument("--json", action="store_true", help="Output in JSON format")

    trace_p = inspect_subs.add_parser("trace-call-chain", help="Greedily trace static call chain from an API endpoint or symbol")
    trace_p.add_argument("--repo", "-r", required=True, help="Target repository")
    trace_p.add_argument("--target", "-t", required=True, help="API endpoint (e.g. /login) or function symbol")
    trace_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    trace_p.add_argument("--max-depth", type=int, default=3, help="Max call stack depth (default: 3)")
    trace_p.add_argument("--token", default=None, help="Personal access token")
    trace_p.add_argument("--json", action="store_true", help="Output in JSON format")

    integrate_p = inspect_subs.add_parser("integrate", help="One-shot microservice integration context (OpenAPI or static call chain)")
    integrate_p.add_argument("--repo", "-r", required=True, help="Target repository")
    integrate_p.add_argument("--target", "-t", required=True, help="API endpoint (e.g. /login) or entity/symbol name")
    integrate_p.add_argument("--ref", default="main", help="Branch or commit SHA")
    integrate_p.add_argument("--subpath", "-s", default=None, help="Optional monorepo subfolder path filter")
    integrate_p.add_argument("--token", default=None, help="Personal access token")
    integrate_p.add_argument("--json", action="store_true", help="Output in JSON format")


def handle_git_remote_inspector_cli(args: argparse.Namespace) -> int:
    service = _create_service()
    ctx = TraceContext(
        trace_id="cli-git-remote-inspector",
        request_id="cli-req-remote-inspect",
        correlation_id="cli-corr-remote-inspect",
    )
    action = getattr(args, "inspect_action", None)

    if action == "discover-contracts":
        cmd = DiscoverContractsCommand(
            repo_url=args.repo,
            ref_name=getattr(args, "ref", "main"),
            token=getattr(args, "token", None),
        )
        res = service.discover_contracts(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Repo: {data['repo_url']} ({data['provider']}) Ref: {data['ref_name']}")
            print(f"Primary Language: {data['primary_language']} | Contracts Found: {data['total_contracts']}")
            for c in data["contracts"]:
                print(f"  [{c['priority']}] {c['category']:<22} {c['path']}")
        return 0

    elif action == "branches":
        cmd = ListBranchesCommand(
            repo_url=args.repo,
            token=getattr(args, "token", None),
        )
        res = service.list_branches(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Repo: {data['repo_url']} | Default: {data['default_branch']} | Total: {data['total_branches']}")
            for b in data["branches"]:
                star = "*" if b["is_default"] else " "
                print(f" {star} {b['name']:<30} {b['commit_sha']}")
        return 0

    elif action == "tree":
        cmd = GetTreeHierarchyCommand(
            repo_url=args.repo,
            ref_name=getattr(args, "ref", "main"),
            max_depth=getattr(args, "max_depth", None),
            token=getattr(args, "token", None),
        )
        res = service.get_tree_hierarchy(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Tree for {data['repo_url']} ({data['ref_name']}) | Files: {data['total_files']} Dirs: {data['total_directories']}")
            print(json.dumps(data["tree_hierarchy"], indent=2))
        return 0

    elif action == "search":
        cmd = SearchFilesCommand(
            repo_url=args.repo,
            ref_name=getattr(args, "ref", "main"),
            path_pattern=getattr(args, "pattern", None),
            token=getattr(args, "token", None),
        )
        res = service.search_files(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Search results for {data['repo_url']} ({data['ref_name']}) | Found: {data['total_found']}")
            for f in data["files"]:
                print(f"  {f['item_type']:<6} {f['path']}")
        return 0

    elif action == "content":
        cmd = GetFileContentCommand(
            repo_url=args.repo,
            file_path=args.path,
            ref_name=getattr(args, "ref", "main"),
            token=getattr(args, "token", None),
        )
        res = service.get_file_content(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(data["content"])
        return 0

    elif action == "diff":
        cmd = GetDiffCommand(
            repo_url=args.repo,
            base_ref=args.base,
            head_ref=args.head,
            token=getattr(args, "token", None),
        )
        res = service.get_diff(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Files Changed: {data['files_changed_count']}")
            print(data["diff_patch"])
        return 0

    elif action == "trace-call-chain":
        cmd = TraceCallChainCommand(
            repo_url=args.repo,
            endpoint_or_symbol=args.target,
            ref_name=getattr(args, "ref", "main"),
            max_depth=getattr(args, "max_depth", 3),
            token=getattr(args, "token", None),
        )
        res = service.trace_call_chain(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            print(f"Target: {data['target_endpoint_or_symbol']} | Entrypoint: {data['entrypoint_file']}")
            print(f"Call Chain (Depth Reached: {data['total_depth_reached']}):")
            for node in data["call_chain"]:
                arrow = "  " * node["depth"] + "->"
                symbols = ", ".join(node["invoked_symbols"][:5])
                print(f"{arrow} [{node['layer']}] {node['file_path']} ({node['symbol_name']})")
                if symbols:
                    print(f"   {'  ' * node['depth']} calls: {symbols}")
        return 0

    elif action == "integrate":
        cmd = IntegrateContextCommand(
            repo_url=args.repo,
            endpoint_or_entity=args.target,
            ref_name=getattr(args, "ref", "main"),
            subpath=getattr(args, "subpath", None),
            token=getattr(args, "token", None),
        )
        res = service.integrate_context(ctx, cmd)
        if res.is_err:
            print(f"Error: {res.error()}", file=sys.stderr)
            return 1
        data = res.unwrap().model_dump()
        if getattr(args, "json", False):
            print(json.dumps(data, indent=2))
        else:
            sig = data["signature"]
            print(f"Integration Context for: {data['target_endpoint_or_entity']} ({data['match_type']})")
            print(f"Repo: {data['repo_url']} ({data['provider']}) | Ref: {data['ref_name']}")
            if data["contract_file"]:
                print(f"Contract Source: {data['contract_file']}")
            print(f"\nProtocol: {sig.get('protocol', 'HTTP_REST')} | Endpoint: {sig['http_method']} {sig['path']}")
            if sig.get("auth_scheme"):
                print(f"Auth Scheme: {sig['auth_scheme']}")
            if sig.get("is_deprecated"):
                print(f"WARNING: This endpoint is marked as DEPRECATED by upstream service.")
            if sig.get("summary"):
                print(f"Summary: {sig['summary']}")
            if sig.get("request_headers"):
                print(f"Headers: {', '.join(sig['request_headers'])}")
            if sig.get("path_params"):
                print(f"Path Params: {', '.join(sig['path_params'])}")
            if sig.get("query_params"):
                print(f"Query Params: {', '.join(sig['query_params'])}")
            if sig.get("sample_request_payload"):
                print(f"\nSample Request Payload:\n{json.dumps(sig['sample_request_payload'], indent=2)}")
            if sig.get("sample_curl"):
                print(f"\nReady-to-Run cURL Command:\n{sig['sample_curl']}")
            if sig.get("client_code_snippet"):
                print(f"\nReady-to-Use Python Client Snippet:\n{sig['client_code_snippet']}")
            if data.get("related_dto_files"):
                print(f"\nRelated DTO / Contract Files:")
                for f in data["related_dto_files"]:
                    print(f"  - {f}")
        return 0

    return 1


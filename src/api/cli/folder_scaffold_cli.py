"""
================================================================================
BLUEPRINT: FOLDER SCAFFOLD CLI DELIVERY ADAPTER
================================================================================
1. OVERVIEW:
   Executes declarative folder tree scaffolding and knowledge graph impact
   analysis via command line. Fully dynamic, consuming shared configuration,
   domain ports, repositories, and centralized architectural constants.
2. SUBCOMMANDS:
   - scaffold-package: Scaffolds complete polyglot sub-package structure
     per 0-package-folder-writing-rule.md using 0-rule-package-structure template.
   - scaffold-feature: Scaffolds backend hexagonal or frontend triad feature
     inside a specific package per 1-feature-folder-writing-rule.md.
   - scaffold-tree: Scaffolds arbitrary custom manifest JSON tree.
   - analyze-impact: Traverses blast radius and impacted callers/callees.
   - analyze-lineage: Traces architectural dependency lineage.
3. ZERO-INLINE-COMMENT DOCTRINE:
   All architecture, argument flows, and parameter mechanics are defined here.
   Method bodies remain 100% comment-free and pure.
================================================================================
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldTreeCommand,
)
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfigLoader
from src.domain.ports.graph_port import GraphStorePort
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
from src.features.folder_scaffold.infrastructure.adapters.folder_scaffold_repository_adapter import AlloyDBFolderScaffoldRepository
from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service import FolderScaffoldAnalyzerDomainService
from src.features.folder_scaffold.contracts.folder_scaffold_deps import FolderScaffoldServiceDeps
from src.shared.constants.scaffold_constants import (
    CLI_CMD_SCAFFOLD_TREE,
    CLI_CMD_SCAFFOLD_PACKAGE,
    CLI_CMD_SCAFFOLD_FEATURE,
    CLI_CMD_SCAFFOLD_DATABASE,
    CLI_CMD_SCAFFOLD_MESSAGING,
    CLI_CMD_SCAFFOLD_DEPLOY,
    CLI_CMD_ANALYZE_IMPACT,
    CLI_CMD_ANALYZE_LINEAGE,
    STORE_MEMORY,
    STORE_NEO4J,
    DIRECTION_UPSTREAM,
    DIRECTION_DOWNSTREAM,
    DEFAULT_TRAVERSAL_DEPTH,
    VAR_PACKAGE_NAME,
    VAR_FEATURE_NAME,
    TRACE_ID_CLI_SCAFFOLD_TREE,
    REQ_ID_CLI_SCAFFOLD_TREE,
    CORR_ID_CLI_SCAFFOLD_TREE,
    TRACE_ID_CLI_ANALYZE_IMPACT,
    REQ_ID_CLI_ANALYZE_IMPACT,
    CORR_ID_CLI_ANALYZE_IMPACT,
    TRACE_ID_CLI_ANALYZE_LINEAGE,
    REQ_ID_CLI_ANALYZE_LINEAGE,
    CORR_ID_CLI_ANALYZE_LINEAGE,
    CFG_KEY_URI,
    CFG_KEY_USER,
    CFG_KEY_PASSWORD,
    CFG_KEY_DATABASE,
    DEFAULT_GRAPH_DATABASE,
    ENCODING_UTF8,
    KEY_NODES,
    KEY_RELATIONSHIPS,
    KEY_PATH,
    KEY_TARGET,
    KEY_SOURCE,
    KEY_PACKAGE_NAME,
    EXIT_SUCCESS,
    EXIT_FAILURE,
    CLI_PROG_FOLDER_SCAFFOLD,
    CLI_DESC_FOLDER_SCAFFOLD,
    CLI_HELP_SCAFFOLD_TREE,
    CLI_HELP_SCAFFOLD_PACKAGE,
    CLI_HELP_SCAFFOLD_FEATURE,
    CLI_HELP_SCAFFOLD_DATABASE,
    CLI_HELP_SCAFFOLD_MESSAGING,
    CLI_HELP_SCAFFOLD_DEPLOY,
    CLI_HELP_ANALYZE_IMPACT,
    CLI_HELP_ANALYZE_LINEAGE,
    CLI_ERR_MANIFEST_REQUIRED,
    CLI_ERR_MANIFEST_NOT_FOUND,
    CLI_ERR_SCAFFOLD_TREE_FAILED,
    CLI_ERR_PACKAGE_NAME_REQUIRED,
    CLI_ERR_FEATURE_NAME_REQUIRED,
    CLI_ERR_TARGET_REQUIRED,
    CLI_ERR_ANALYZE_IMPACT_FAILED,
    CLI_ERR_FILE_REQUIRED,
    CLI_ERR_ANALYZE_LINEAGE_FAILED,
    TEMPLATE_FILE_PACKAGE_STRUCTURE,
    TEMPLATE_FILE_BACKEND_FEATURE,
    TEMPLATE_FILE_FRONTEND_FEATURE,
    TEMPLATE_FILE_DATABASE_STRUCTURE,
    TEMPLATE_FILE_MESSAGING_STRUCTURE,
    TEMPLATE_FILE_DEPLOY_STRUCTURE,
    FEATURE_TYPE_BACKEND,
    FEATURE_TYPE_FRONTEND,
    HEADER_TREE_SCAFFOLDED,
    HEADER_BLAST_RADIUS,
    HEADER_FILE_LINEAGE,
)


def get_template_path(template_name: str) -> Path:
    return Path(__file__).resolve().parent.parent.parent / "features" / "folder_scaffold" / "templates" / template_name


def resolve_graph_store(args: argparse.Namespace) -> GraphStorePort:
    store = getattr(args, "store", None)
    neo4j_flag = getattr(args, "neo4j", False)
    if neo4j_flag or store == STORE_NEO4J:
        kg_cfg = AppConfigLoader.get_knowledge_graph_config()
        return Neo4jGraphAdapter(
            uri=kg_cfg.get(CFG_KEY_URI),
            user=kg_cfg.get(CFG_KEY_USER),
            password=kg_cfg.get(CFG_KEY_PASSWORD),
            database=kg_cfg.get(CFG_KEY_DATABASE, DEFAULT_GRAPH_DATABASE),
        )
    return InMemoryGraphAdapter()


def create_folder_scaffold_service(args: argparse.Namespace) -> FolderScaffoldDomainService:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBFolderScaffoldRepository(db_url=db_url)
    graph_store = resolve_graph_store(args)
    scaffold_cfg = AppConfigLoader.get_scaffold_config()
    deps = FolderScaffoldServiceDeps(
        repository=repo,
        graph_store=graph_store,
        scaffold_config=scaffold_cfg,
    )
    return FolderScaffoldDomainService(deps=deps)


def create_folder_scaffold_analyzer(args: argparse.Namespace) -> FolderScaffoldAnalyzerDomainService:
    db_url = AppConfigLoader.get_database_url()
    repo = AlloyDBFolderScaffoldRepository(db_url=db_url)
    graph_store = resolve_graph_store(args)
    return FolderScaffoldAnalyzerDomainService(repository=repo, graph_store=graph_store)


def execute_scaffold_manifest(
    manifest_path: Path,
    base_dir: Optional[str],
    package_name: Optional[str],
    feature_name: Optional[str],
    root_id: Optional[str],
    args: argparse.Namespace,
) -> int:
    if not manifest_path.exists():
        print(CLI_ERR_MANIFEST_NOT_FOUND.format(path=str(manifest_path)))
        return EXIT_FAILURE

    try:
        raw_text = manifest_path.read_text(encoding=ENCODING_UTF8)
        if package_name:
            raw_text = raw_text.replace(VAR_PACKAGE_NAME, package_name)
        if feature_name:
            raw_text = raw_text.replace(VAR_FEATURE_NAME, feature_name)
        if root_id:
            raw_text = raw_text.replace(VAR_PACKAGE_NAME, root_id).replace(VAR_FEATURE_NAME, root_id)

        raw_payload = json.loads(raw_text)
        if base_dir:
            base_path = Path(base_dir)
            base_path.mkdir(parents=True, exist_ok=True)
            for node in raw_payload.get(KEY_NODES, []):
                if not Path(node[KEY_PATH]).is_absolute():
                    node[KEY_PATH] = str(base_path / node[KEY_PATH])
                target_val = node.get(KEY_TARGET, "")
                if "/" in target_val or "." in target_val:
                    if not Path(target_val).is_absolute():
                        node[KEY_TARGET] = str(base_path / target_val)
            for rel in raw_payload.get(KEY_RELATIONSHIPS, []):
                src_val = rel.get(KEY_SOURCE, "")
                if "/" in src_val or "." in src_val:
                    if not Path(src_val).is_absolute():
                        rel[KEY_SOURCE] = str(base_path / src_val)
                tgt_val = rel.get(KEY_TARGET, "")
                if "/" in tgt_val or "." in tgt_val:
                    if not Path(tgt_val).is_absolute():
                        rel[KEY_TARGET] = str(base_path / tgt_val)

        if package_name:
            for node in raw_payload.get(KEY_NODES, []):
                if not node.get(KEY_PACKAGE_NAME) or node.get(KEY_PACKAGE_NAME) == VAR_PACKAGE_NAME:
                    node[KEY_PACKAGE_NAME] = package_name

        cmd = ScaffoldTreeCommand.model_validate(raw_payload)
        service = create_folder_scaffold_service(args)
        ctx = TraceContext(
            trace_id=TRACE_ID_CLI_SCAFFOLD_TREE,
            request_id=REQ_ID_CLI_SCAFFOLD_TREE,
            correlation_id=CORR_ID_CLI_SCAFFOLD_TREE,
        )
        result = service.scaffoldTree(ctx, cmd)
        if result.is_err:
            print(CLI_ERR_SCAFFOLD_TREE_FAILED.format(error=result.error()))
            return EXIT_FAILURE

        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("=" * 80)
            print(HEADER_TREE_SCAFFOLDED)
            print("=" * 80)
            print(f"Base Path:             {cmd.root_id}")
            print(f"Nodes Created:         {len(res.nodes)}")
            print(f"Relationships Linked:  {len(res.relationships)}")
            print(f"Duration:              {res.duration_ms} ms")
            print(f"Status:                {res.status}")
            print("=" * 80)
        return EXIT_SUCCESS
    except Exception as exc:
        print(CLI_ERR_SCAFFOLD_TREE_FAILED.format(error=exc))
        return EXIT_FAILURE


def handle_scaffold_package(args: argparse.Namespace) -> int:
    package_name = getattr(args, "package_name", None) or getattr(args, "name", None)
    if not package_name:
        print(CLI_ERR_PACKAGE_NAME_REQUIRED)
        return EXIT_FAILURE

    base_dir = getattr(args, "base_dir", None)
    if not base_dir:
        base_dir = str(Path(repo_root).parent.parent / "packages" / package_name)

    manifest_path = get_template_path(TEMPLATE_FILE_PACKAGE_STRUCTURE)
    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=package_name,
        feature_name=None,
        root_id=package_name,
        args=args,
    )


def handle_scaffold_feature(args: argparse.Namespace) -> int:
    feature_name = getattr(args, "feature_name", None) or getattr(args, "name", None)
    package_name = getattr(args, "package_name", None) or getattr(args, "package", None)
    if not feature_name or not package_name:
        print(CLI_ERR_FEATURE_NAME_REQUIRED)
        return EXIT_FAILURE

    feature_type = getattr(args, "feature_type", FEATURE_TYPE_BACKEND)
    template_name = (
        TEMPLATE_FILE_FRONTEND_FEATURE
        if feature_type == FEATURE_TYPE_FRONTEND
        else TEMPLATE_FILE_BACKEND_FEATURE
    )
    manifest_path = get_template_path(template_name)

    base_dir = getattr(args, "base_dir", None)
    if not base_dir:
        base_dir = str(Path(repo_root).parent.parent / "packages" / package_name)

    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=package_name,
        feature_name=feature_name,
        root_id=f"{package_name}/{feature_name}",
        args=args,
    )


def handle_scaffold_database(args: argparse.Namespace) -> int:
    package_name = getattr(args, "package_name", None) or getattr(args, "package", None) or getattr(args, "name", None)
    if not package_name:
        print(CLI_ERR_PACKAGE_NAME_REQUIRED)
        return EXIT_FAILURE

    manifest_path = get_template_path(TEMPLATE_FILE_DATABASE_STRUCTURE)
    base_dir = getattr(args, "base_dir", None)
    if not base_dir:
        base_dir = str(Path(repo_root).parent.parent / "packages" / package_name)

    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=package_name,
        feature_name=None,
        root_id=f"{package_name}/database",
        args=args,
    )


def handle_scaffold_messaging(args: argparse.Namespace) -> int:
    package_name = getattr(args, "package_name", None) or getattr(args, "package", None) or getattr(args, "name", None)
    if not package_name:
        print(CLI_ERR_PACKAGE_NAME_REQUIRED)
        return EXIT_FAILURE

    manifest_path = get_template_path(TEMPLATE_FILE_MESSAGING_STRUCTURE)
    base_dir = getattr(args, "base_dir", None)
    if not base_dir:
        base_dir = str(Path(repo_root).parent.parent / "packages" / package_name)

    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=package_name,
        feature_name=None,
        root_id=f"{package_name}/messaging",
        args=args,
    )


def handle_scaffold_deploy(args: argparse.Namespace) -> int:
    package_name = getattr(args, "package_name", None) or getattr(args, "package", None) or getattr(args, "name", None)
    if not package_name:
        print(CLI_ERR_PACKAGE_NAME_REQUIRED)
        return EXIT_FAILURE

    manifest_path = get_template_path(TEMPLATE_FILE_DEPLOY_STRUCTURE)
    base_dir = getattr(args, "base_dir", None)
    if not base_dir:
        base_dir = str(Path(repo_root).parent.parent / "packages" / package_name)

    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=package_name,
        feature_name=None,
        root_id=f"{package_name}/deploy",
        args=args,
    )


def handle_scaffold_tree(args: argparse.Namespace) -> int:
    manifest_arg = getattr(args, "manifest", None)
    if not manifest_arg:
        print(CLI_ERR_MANIFEST_REQUIRED)
        return EXIT_FAILURE

    manifest_path = Path(manifest_arg)
    pkg_name = getattr(args, "package_name", None) or getattr(args, "package", None)
    feat_name = getattr(args, "feature_name", None) or getattr(args, "feature", None)
    root_id_arg = getattr(args, "root_id", None)
    base_dir = getattr(args, "base_dir", None)

    return execute_scaffold_manifest(
        manifest_path=manifest_path,
        base_dir=base_dir,
        package_name=pkg_name,
        feature_name=feat_name,
        root_id=root_id_arg,
        args=args,
    )


def handle_analyze_impact(args: argparse.Namespace) -> int:
    target_id = getattr(args, "target", None) or getattr(args, "target_id", None)
    direction = getattr(args, "direction", DIRECTION_UPSTREAM)
    max_depth = getattr(args, "depth", DEFAULT_TRAVERSAL_DEPTH)

    if not target_id:
        print(CLI_ERR_TARGET_REQUIRED)
        return EXIT_FAILURE

    try:
        analyzer = create_folder_scaffold_analyzer(args)
        ctx = TraceContext(
            trace_id=TRACE_ID_CLI_ANALYZE_IMPACT,
            request_id=REQ_ID_CLI_ANALYZE_IMPACT,
            correlation_id=CORR_ID_CLI_ANALYZE_IMPACT,
        )
        result = analyzer.analyze_impact(
            ctx=ctx,
            target_id=target_id,
            direction=direction,
            max_depth=int(max_depth),
        )
        if result.is_err:
            print(CLI_ERR_ANALYZE_IMPACT_FAILED.format(error=result.error()))
            return EXIT_FAILURE

        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("=" * 80)
            print(HEADER_BLAST_RADIUS)
            print("=" * 80)
            print(f"Target Node:       {res.target_id}")
            print(f"Direction:         {res.direction}")
            print(f"Max Depth:         {res.max_depth}")
            print(f"Total Impacted:    {res.total_impacted}")
            print(f"Risk Level:        {res.risk_level}")
            print(f"Blast Score:       {res.blast_radius_score}")
            print("-" * 80)
            for node in res.impacted_nodes:
                print(f" - [{node.depth} hop(s)] {node.file_path or node.id} (via :{node.relationship}) [{node.risk_factor}]")
            print("=" * 80)
        return EXIT_SUCCESS
    except Exception as exc:
        print(CLI_ERR_ANALYZE_IMPACT_FAILED.format(error=exc))
        return EXIT_FAILURE


def handle_analyze_lineage(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None) or getattr(args, "file_path", None)
    max_depth = getattr(args, "depth", DEFAULT_TRAVERSAL_DEPTH)

    if not file_path:
        print(CLI_ERR_FILE_REQUIRED)
        return EXIT_FAILURE

    try:
        analyzer = create_folder_scaffold_analyzer(args)
        ctx = TraceContext(
            trace_id=TRACE_ID_CLI_ANALYZE_LINEAGE,
            request_id=REQ_ID_CLI_ANALYZE_LINEAGE,
            correlation_id=CORR_ID_CLI_ANALYZE_LINEAGE,
        )
        result = analyzer.get_file_lineage(
            ctx=ctx,
            file_path=file_path,
            max_depth=int(max_depth),
        )
        if result.is_err:
            print(CLI_ERR_ANALYZE_LINEAGE_FAILED.format(error=result.error()))
            return EXIT_FAILURE

        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("=" * 80)
            print(HEADER_FILE_LINEAGE)
            print("=" * 80)
            print(f"File Path:     {res.file_path}")
            print(f"Role:          {res.role}")
            print(f"Layer:         {res.layer}")
            print(f"Direct Up:     {len(res.direct_upstream)}")
            print(f"Direct Down:   {len(res.direct_downstream)}")
            print(f"Blast Score:   {res.blast_radius_score}")
            print("-" * 80)
            print("Direct Dependencies (Upstream Callers):")
            for dep in res.direct_upstream:
                print(f"  <- {dep.get('file_path')} via :{dep.get('rel_type')}")
            print("Direct Dependents (Downstream Callee):")
            for dep in res.direct_downstream:
                print(f"  -> {dep.get('file_path')} via :{dep.get('rel_type')}")
            print("=" * 80)
        return EXIT_SUCCESS
    except Exception as exc:
        print(CLI_ERR_ANALYZE_LINEAGE_FAILED.format(error=exc))
        return EXIT_FAILURE


def build_folder_scaffold_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=CLI_PROG_FOLDER_SCAFFOLD,
        description=CLI_DESC_FOLDER_SCAFFOLD,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    scaffold_pkg_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_PACKAGE,
        help=CLI_HELP_SCAFFOLD_PACKAGE,
    )
    scaffold_pkg_p.add_argument("--name", "--package-name", dest="name", required=True, help="Sub-package name to scaffold")
    scaffold_pkg_p.add_argument("--base-dir", dest="base_dir", default=None, help="Root base directory (default: packages/<name>)")
    scaffold_pkg_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_pkg_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_pkg_p.add_argument("--json", action="store_true", help="Output result as JSON")

    scaffold_feat_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_FEATURE,
        help=CLI_HELP_SCAFFOLD_FEATURE,
    )
    scaffold_feat_p.add_argument("--name", "--feature-name", dest="name", required=True, help="Feature name to scaffold")
    scaffold_feat_p.add_argument("--package", "--package-name", dest="package", required=True, help="Target package name where feature resides")
    scaffold_feat_p.add_argument("--type", dest="feature_type", choices=[FEATURE_TYPE_BACKEND, FEATURE_TYPE_FRONTEND], default=FEATURE_TYPE_BACKEND, help="Feature architecture paradigm (backend or frontend)")
    scaffold_feat_p.add_argument("--base-dir", dest="base_dir", default=None, help="Package root directory (default: packages/<package>)")
    scaffold_feat_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_feat_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_feat_p.add_argument("--json", action="store_true", help="Output result as JSON")

    scaffold_tree_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_TREE,
        help=CLI_HELP_SCAFFOLD_TREE,
    )
    scaffold_tree_p.add_argument("--manifest", required=True, help="Path to JSON manifest defining nodes and relationships")
    scaffold_tree_p.add_argument("--base-dir", dest="base_dir", default=None, help="Base directory where package/feature should be scaffolded")
    scaffold_tree_p.add_argument("--package", "--package-name", dest="package_name", default=None, help="Package name variable substitution")
    scaffold_tree_p.add_argument("--feature", "--feature-name", dest="feature_name", default=None, help="Feature name variable substitution")
    scaffold_tree_p.add_argument("--root-id", dest="root_id", default=None, help="Root identifier substitution")
    scaffold_tree_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_tree_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_tree_p.add_argument("--json", action="store_true", help="Output result as JSON")

    impact_p = subparsers.add_parser(
        CLI_CMD_ANALYZE_IMPACT,
        help=CLI_HELP_ANALYZE_IMPACT,
    )
    impact_p.add_argument("--target", "--target-id", dest="target", required=True, help="Target node ID or file path")
    impact_p.add_argument("--direction", choices=[DIRECTION_UPSTREAM, DIRECTION_DOWNSTREAM], default=DIRECTION_UPSTREAM, help="Traversal direction")
    impact_p.add_argument("--depth", type=int, default=DEFAULT_TRAVERSAL_DEPTH, help="Max traversal depth in graph")
    impact_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    impact_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    impact_p.add_argument("--json", action="store_true", help="Output result as JSON")

    lineage_p = subparsers.add_parser(
        CLI_CMD_ANALYZE_LINEAGE,
        help=CLI_HELP_ANALYZE_LINEAGE,
    )
    lineage_p.add_argument("--file", "--path", dest="file", required=True, help="File path to trace lineage for")
    lineage_p.add_argument("--depth", type=int, default=DEFAULT_TRAVERSAL_DEPTH, help="Max lineage traversal depth")
    lineage_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    lineage_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    lineage_p.add_argument("--json", action="store_true", help="Output result as JSON")

    scaffold_db_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_DATABASE,
        help=CLI_HELP_SCAFFOLD_DATABASE,
    )
    scaffold_db_p.add_argument("--package", "--package-name", "--name", dest="package", required=True, help="Target package name where database hub should reside")
    scaffold_db_p.add_argument("--base-dir", dest="base_dir", default=None, help="Package root directory (default: packages/<package>)")
    scaffold_db_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_db_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_db_p.add_argument("--json", action="store_true", help="Output result as JSON")

    scaffold_msg_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_MESSAGING,
        help=CLI_HELP_SCAFFOLD_MESSAGING,
    )
    scaffold_msg_p.add_argument("--package", "--package-name", "--name", dest="package", required=True, help="Target package name where messaging hub should reside")
    scaffold_msg_p.add_argument("--base-dir", dest="base_dir", default=None, help="Package root directory (default: packages/<package>)")
    scaffold_msg_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_msg_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_msg_p.add_argument("--json", action="store_true", help="Output result as JSON")

    scaffold_deploy_p = subparsers.add_parser(
        CLI_CMD_SCAFFOLD_DEPLOY,
        help=CLI_HELP_SCAFFOLD_DEPLOY,
    )
    scaffold_deploy_p.add_argument("--package", "--package-name", "--name", dest="package", required=True, help="Target package name where deploy hub should reside")
    scaffold_deploy_p.add_argument("--base-dir", dest="base_dir", default=None, help="Package root directory (default: packages/<package>)")
    scaffold_deploy_p.add_argument("--store", choices=[STORE_MEMORY, STORE_NEO4J], default=STORE_MEMORY, help="Graph storage backend")
    scaffold_deploy_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_deploy_p.add_argument("--json", action="store_true", help="Output result as JSON")

    return parser


def main() -> None:
    parser = build_folder_scaffold_cli_parser()
    args = parser.parse_args()
    dispatch = {
        CLI_CMD_SCAFFOLD_PACKAGE: handle_scaffold_package,
        CLI_CMD_SCAFFOLD_FEATURE: handle_scaffold_feature,
        CLI_CMD_SCAFFOLD_DATABASE: handle_scaffold_database,
        CLI_CMD_SCAFFOLD_MESSAGING: handle_scaffold_messaging,
        CLI_CMD_SCAFFOLD_DEPLOY: handle_scaffold_deploy,
        CLI_CMD_SCAFFOLD_TREE: handle_scaffold_tree,
        CLI_CMD_ANALYZE_IMPACT: handle_analyze_impact,
        CLI_CMD_ANALYZE_LINEAGE: handle_analyze_lineage,
    }
    handler = dispatch.get(args.subcommand)
    if handler:
        sys.exit(handler(args))


if __name__ == "__main__":
    main()

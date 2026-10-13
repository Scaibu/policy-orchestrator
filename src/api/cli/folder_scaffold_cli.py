"""
Folder Scaffold CLI Delivery Adapter
Executes declarative folder tree scaffolding and knowledge graph impact analysis via command line.
"""
import argparse
import json
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    ScaffoldTreeCommand,
)
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfigLoader
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
from src.features.folder_scaffold.repository.folder_scaffold_repository import AlloyDBFolderScaffoldRepository
from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service import FolderScaffoldAnalyzerDomainService
from src.features.folder_scaffold.types.folder_scaffold_deps import FolderScaffoldServiceDeps


def resolve_graph_store(args: argparse.Namespace):
    store = getattr(args, "store", None)
    neo4j_flag = getattr(args, "neo4j", False)
    if neo4j_flag or store == "neo4j":
        kg_cfg = AppConfigLoader.get_knowledge_graph_config()
        return Neo4jGraphAdapter(
            uri=kg_cfg.get("uri"),
            user=kg_cfg.get("user"),
            password=kg_cfg.get("password"),
            database=kg_cfg.get("database", "neo4j"),
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


def handle_scaffold_tree(args: argparse.Namespace) -> int:
    manifest_arg = getattr(args, "manifest", None)
    if not manifest_arg:
        print("Error: --manifest <path.json> is required for scaffold-tree.")
        return 1

    manifest_path = Path(manifest_arg)
    if not manifest_path.exists():
        print(f"Error: Manifest file '{manifest_arg}' does not exist.")
        return 1

    try:
        raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        cmd = ScaffoldTreeCommand.model_validate(raw_payload)
        service = create_folder_scaffold_service(args)
        ctx = TraceContext(
            trace_id="cli-scaffold-tree",
            request_id="cli-req-scaffold-tree",
            correlation_id="cli-corr-scaffold-tree",
        )
        result = service.scaffoldTree(ctx, cmd)
        if result.is_err:
            print(f"Error scaffolding tree: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("================================================================================")
            print("TREE SCAFFOLDED & KNOWLEDGE GRAPH SYNCHRONIZED")
            print("================================================================================")
            print(f"Base Path:             {cmd.base_path}")
            print(f"Nodes Created:         {len(res.nodes_created)}")
            print(f"Relationships Linked:  {len(res.relationships_linked)}")
            print(f"Duration:              {res.duration_ms} ms")
            print(f"Status:                {res.status}")
            print("================================================================================")
        return 0
    except Exception as exc:
        print(f"Error scaffolding tree: {exc}")
        return 1


def handle_analyze_impact(args: argparse.Namespace) -> int:
    target_id = getattr(args, "target", None) or getattr(args, "target_id", None)
    direction = getattr(args, "direction", "upstream")
    max_depth = getattr(args, "depth", 5)

    if not target_id:
        print("Error: --target <node_id> is required for impact analysis.")
        return 1

    try:
        analyzer = create_folder_scaffold_analyzer(args)
        ctx = TraceContext(
            trace_id="cli-analyze-impact",
            request_id="cli-req-analyze-impact",
            correlation_id="cli-corr-analyze-impact",
        )
        result = analyzer.analyze_impact(
            ctx=ctx,
            target_id=target_id,
            direction=direction,
            max_depth=int(max_depth),
        )
        if result.is_err:
            print(f"Error analyzing impact: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("================================================================================")
            print("BLAST RADIUS & GRAPH IMPACT ANALYSIS")
            print("================================================================================")
            print(f"Target Node:       {res.target_id}")
            print(f"Direction:         {res.direction}")
            print(f"Max Depth:         {res.traversal_depth}")
            print(f"Total Impacted:    {res.impacted_count}")
            print("--------------------------------------------------------------------------------")
            for node in res.impacted_nodes:
                print(f" - [{node.depth} hop(s)] {node.file_path or node.node_id} (via :{node.relationship_type})")
            print("================================================================================")
        return 0
    except Exception as exc:
        print(f"Error analyzing impact: {exc}")
        return 1


def handle_analyze_lineage(args: argparse.Namespace) -> int:
    file_path = getattr(args, "file", None) or getattr(args, "file_path", None)
    max_depth = getattr(args, "depth", 5)

    if not file_path:
        print("Error: --file <file_path> is required for file lineage tracing.")
        return 1

    try:
        analyzer = create_folder_scaffold_analyzer(args)
        ctx = TraceContext(
            trace_id="cli-analyze-lineage",
            request_id="cli-req-analyze-lineage",
            correlation_id="cli-corr-analyze-lineage",
        )
        result = analyzer.get_file_lineage(
            ctx=ctx,
            file_path=file_path,
            max_depth=int(max_depth),
        )
        if result.is_err:
            print(f"Error tracing lineage: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("================================================================================")
            print("FILE KNOWLEDGE GRAPH LINEAGE")
            print("================================================================================")
            print(f"File Path:     {res.file_path}")
            print(f"Direct Deps:   {len(res.direct_dependencies)}")
            print(f"Transitive:    {len(res.transitive_dependencies)}")
            print("--------------------------------------------------------------------------------")
            print("Direct Dependencies:")
            for dep in res.direct_dependencies:
                print(f"  -> {dep}")
            print("Transitive Dependencies:")
            for dep in res.transitive_dependencies:
                print(f"  *  {dep}")
            print("================================================================================")
        return 0
    except Exception as exc:
        print(f"Error tracing lineage: {exc}")
        return 1


def build_folder_scaffold_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folder-scaffold-cli",
        description="Declarative Folder Scaffold & Graph Analysis CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    scaffold_tree_p = subparsers.add_parser(
        "scaffold-tree",
        help="Declaratively scaffold a tree of folders, files, and relationships from a JSON manifest",
    )
    scaffold_tree_p.add_argument("--manifest", required=True, help="Path to JSON manifest defining nodes and relationships")
    scaffold_tree_p.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="Graph storage backend")
    scaffold_tree_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    scaffold_tree_p.add_argument("--json", action="store_true", help="Output result as JSON")

    impact_p = subparsers.add_parser(
        "analyze-impact",
        help="Analyze blast radius / impacted nodes for a target file or node in the knowledge graph",
    )
    impact_p.add_argument("--target", "--target-id", dest="target", required=True, help="Target node ID or file path")
    impact_p.add_argument("--direction", choices=["upstream", "downstream"], default="upstream", help="Traversal direction")
    impact_p.add_argument("--depth", type=int, default=5, help="Max traversal depth in graph")
    impact_p.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="Graph storage backend")
    impact_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    impact_p.add_argument("--json", action="store_true", help="Output result as JSON")

    lineage_p = subparsers.add_parser(
        "analyze-lineage",
        help="Trace architectural dependency lineage for a given file in the knowledge graph",
    )
    lineage_p.add_argument("--file", "--path", dest="file", required=True, help="File path to trace lineage for")
    lineage_p.add_argument("--depth", type=int, default=5, help="Max lineage traversal depth")
    lineage_p.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="Graph storage backend")
    lineage_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    lineage_p.add_argument("--json", action="store_true", help="Output result as JSON")

    return parser


def main() -> None:
    parser = build_folder_scaffold_cli_parser()
    args = parser.parse_args()
    dispatch = {
        "scaffold-tree": handle_scaffold_tree,
        "analyze-impact": handle_analyze_impact,
        "analyze-lineage": handle_analyze_lineage,
    }
    handler = dispatch.get(args.subcommand)
    if handler:
        sys.exit(handler(args))


if __name__ == "__main__":
    main()

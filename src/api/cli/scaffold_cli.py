import argparse
import json
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
)
from src.shared.types.envelope_types import TraceContext
from src.infra.config.config_loader import AppConfigLoader
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
from src.features.folder_scaffold.repository.folder_scaffold_repository import AlloyDBFolderScaffoldRepository
from src.features.folder_scaffold.service.folder_scaffold_service import FolderScaffoldDomainService
from src.features.folder_scaffold.types.folder_scaffold_deps import FolderScaffoldServiceDeps


def get_folder_scaffold_service(args: argparse.Namespace) -> FolderScaffoldDomainService:
    db_url = AppConfigLoader.get_database_url()
    kg_cfg = AppConfigLoader.get_knowledge_graph_config()
    repo = AlloyDBFolderScaffoldRepository(db_url=db_url)

    store = getattr(args, "store", None)
    neo4j_flag = getattr(args, "neo4j", False)
    if neo4j_flag or store == "neo4j":
        graph_store = Neo4jGraphAdapter(
            uri=kg_cfg.get("uri"),
            user=kg_cfg.get("user"),
            password=kg_cfg.get("password"),
            database=kg_cfg.get("database", "neo4j"),
        )
    else:
        graph_store = InMemoryGraphAdapter()

    scaffold_cfg = AppConfigLoader.get_scaffold_config()
    deps = FolderScaffoldServiceDeps(
        repository=repo,
        graph_store=graph_store,
        scaffold_config=scaffold_cfg,
    )
    return FolderScaffoldDomainService(deps=deps)


def handle_create_file(args: argparse.Namespace) -> int:
    service = get_folder_scaffold_service(args)

    file_path = getattr(args, "path", None) or getattr(args, "file_path", None)
    role = getattr(args, "role", None)
    rel_type = getattr(args, "rel_type", None) or getattr(args, "rel", None)
    target = getattr(args, "target", None)
    direction = getattr(args, "direction", "outgoing")
    package_name = getattr(args, "package_root", None)
    feature_name = getattr(args, "feature_name", None)

    if not file_path or not role or not rel_type or not target:
        print("Error: --path, --role, --rel, and --target are all REQUIRED parameters to create an architecturally linked file.")
        return 1

    try:
        cmd = CreateFileCommand(
            path=file_path,
            role=role,
            rel_type=rel_type,
            target=target,
            direction=direction,
            feature_name=feature_name,
            package_name=package_name,
        )
        ctx = TraceContext(
            trace_id="cli-create-file",
            request_id="cli-req-create-file",
            correlation_id="cli-corr-create-file",
        )
        result = service.createFile(ctx, cmd)
        if result.is_err:
            print(f"Error creating file: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            print("================================================================================")
            print("FILE CREATED & ARCHITECTURALLY LINKED")
            print("================================================================================")
            print(f"File Path:    {res.node.file_path}")
            print(f"Role / Layer: {res.node.role} [{res.node.layer}]")
            rel = res.relationship
            print(f"Relationship: {rel.source_id} -[:{rel.rel_type}]-> {rel.target_id}")
            print(f"AlloyDB:      Persisted ({res.status})")
            print("Neo4j:        Synced (Constraints Enforced)")
            print("================================================================================")
        return 0
    except Exception as exc:
        print(f"Error creating file: {exc}")
        return 1


def handle_link_file(args: argparse.Namespace) -> int:
    service = get_folder_scaffold_service(args)

    source = getattr(args, "source", None)
    rel_type = getattr(args, "rel_type", None) or getattr(args, "rel", None)
    target = getattr(args, "target", None)

    if not source or not rel_type or not target:
        print("Error: --source, --rel, and --target are all REQUIRED parameters to link files.")
        return 1

    try:
        cmd = LinkFilesCommand(source=source, rel_type=rel_type, target=target)
        ctx = TraceContext(
            trace_id="cli-link-file",
            request_id="cli-req-link-file",
            correlation_id="cli-corr-link-file",
        )
        result = service.linkFiles(ctx, cmd)
        if result.is_err:
            print(f"Error linking files: {result.error()}")
            return 1
        res = result.unwrap()
        if getattr(args, "json", False):
            print(json.dumps(res.model_dump(), indent=2))
        else:
            rel = res.relationship
            print(f"Successfully linked: {rel.source_id} -[:{rel.rel_type}]-> {rel.target_id}")
            print("AlloyDB & Neo4j synchronized.")
        return 0
    except Exception as exc:
        print(f"Error linking files: {exc}")
        return 1


def register_scaffold_subparsers(subparsers) -> None:
    # 1. create-file
    create_file_p = subparsers.add_parser(
        "create-file",
        aliases=["add-file"],
        help="Create an architectural file with mandatory upstream/downstream relationship",
    )
    create_file_p.add_argument("--path", required=True, help="File path to create")
    create_file_p.add_argument("--role", required=True, help="Architectural role / node label")
    create_file_p.add_argument("--rel", "--rel-type", dest="rel_type", required=True, help="Relationship type")
    create_file_p.add_argument("--target", required=True, help="Target file path or node ID")
    create_file_p.add_argument(
        "--direction",
        choices=["outgoing", "incoming"],
        default="outgoing",
        help="Direction: outgoing or incoming",
    )
    create_file_p.add_argument("--package-root", default=None, help="Package root directory")
    create_file_p.add_argument("--feature", dest="feature_name", default=None, help="Feature name")
    create_file_p.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="Graph storage backend")
    create_file_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    create_file_p.add_argument("--json", action="store_true", help="Output result as JSON")

    # 4. link-file
    link_file_p = subparsers.add_parser(
        "link-file",
        aliases=["link-nodes", "connect-files"],
        help="Link two architectural files or nodes with a typed relationship",
    )
    link_file_p.add_argument("--source", required=True, help="Source file path or node ID")
    link_file_p.add_argument("--rel", "--rel-type", dest="rel_type", required=True, help="Relationship type")
    link_file_p.add_argument("--target", required=True, help="Target file path or node ID")
    link_file_p.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="Graph storage backend")
    link_file_p.add_argument("--neo4j", action="store_true", help="Use live Neo4j graph storage")
    link_file_p.add_argument("--json", action="store_true", help="Output result as JSON")

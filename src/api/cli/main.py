#!/usr/bin/env python3
"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: ERGONOMIC POLICY ORCHESTRATOR CLI
================================================================================

1. OVERVIEW & OBJECTIVE:
   Provides a clean, intuitive CLI for database migrations, folder scaffolding,
   and FastAPI server management.

2. SUBCOMMANDS:
   - `migrate`: Runs/rolls back SQL & Neo4j graph migrations.
   - `create-file`: Scaffolds architecturally linked files.
   - `link-file`: Links nodes with graph relationships.
   - `serve`: Launches FastAPI REST server via Uvicorn.

3. ARCHITECTURAL INVARIANTS:
   - Zero-Inline-Comment Doctrine: 100% pure method/function bodies.
   - Standardized Posix Exit Codes: 0 on success, 1 on error/violation.
================================================================================
"""

import sys
import os
import argparse
import json
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.infra.adapters.database.migration_runner import DatabaseMigrationRunner


def handle_migrate_command(args: argparse.Namespace) -> int:
    from src.infra.config.config_loader import AppConfigLoader
    db_target = getattr(args, "db", "all")
    if db_target == "sqlite":
        db_url = f"sqlite:///{getattr(args, 'sqlite_path', 'policy_registry.db')}"
    else:
        db_url = getattr(args, "postgres_url", None) or AppConfigLoader.get_database_url()

    runner = DatabaseMigrationRunner(db_url)
    action = getattr(args, "action", "run")

    if action == "run":
        if db_target in ("all", "alloydb", "postgres"):
            runner.run_migrations()
            print("✅ SQL migrations applied successfully.")
        if db_target in ("all", "neo4j"):
            graph_res = runner.run_graph_migrations()
            print(f"✅ Neo4j Graph constraint migration successful: {graph_res}")
        return 0
    elif action == "rollback":
        if db_target in ("all", "neo4j"):
            graph_res = runner.rollback_graph_migrations()
            print(f"🔄 Neo4j Graph rollback successful: {graph_res}")
        if db_target in ("all", "alloydb", "postgres", "sqlite"):
            res = runner.rollback_migrations()
            print(f"🔄 SQL rollback successful: {res}")
        return 0
    elif action == "status":
        status = runner.get_migration_status()
        print(json.dumps(status, indent=2))
        return 0
    return 0


def handle_serve_command(args: argparse.Namespace) -> int:
    import uvicorn
    os.environ["POLICY_RULES_DIR"] = args.rules_dir
    os.environ["LLM_BACKEND"] = args.backend
    uvicorn.run("src.api.rest.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="policy-orchestrator",
        description="Policy Orchestrator CLI for migrations, scaffolding, and service hosting",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("action", choices=["run", "status", "rollback"], default="run", nargs="?", help="Migration action")
    migrate_parser.add_argument("--db", choices=["all", "alloydb", "postgres", "neo4j", "sqlite"], default="all", help="Target database")
    migrate_parser.add_argument("--sqlite-path", default="policy_registry.db", help="SQLite database path")
    migrate_parser.add_argument("--postgres-url", default=None, help="PostgreSQL connection string")

    from src.api.cli.scaffold_cli import register_scaffold_subparsers
    register_scaffold_subparsers(subparsers)

    serve_parser = subparsers.add_parser("serve", help="Launch FastAPI REST server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.add_argument("--rules-dir", default="../rules", help="Path to rules folder")
    serve_parser.add_argument("--backend", default="mock", choices=["mock", "openai", "ollama"], help="LLM backend")

    return parser


def main() -> None:
    from src.api.cli.scaffold_cli import (
        handle_create_file,
        handle_link_file,
    )
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "migrate": handle_migrate_command,
        "create-file": handle_create_file,
        "add-file": handle_create_file,
        "link-file": handle_link_file,
        "link-nodes": handle_link_file,
        "connect-files": handle_link_file,
        "serve": handle_serve_command,
    }

    handler = dispatch.get(args.subcommand)
    if handler:
        sys.exit(handler(args))


if __name__ == "__main__":
    main()

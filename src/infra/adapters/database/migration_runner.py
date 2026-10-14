"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: DATABASE MIGRATION & SEED RUNNER
================================================================================

1. OVERVIEW & OBJECTIVE:
   Automated database schema migration, rollback, status tracking, and
   algorithm catalog seeder supporting Google AlloyDB Omni, PostgreSQL, and SQLite.
   Discovers and executes SQL migration scripts from database/migrations/
   with idempotent schema ledger tracking and contract parity verification.

2. ARCHITECTURAL LOCATION:
   src/infra/adapters/database/migration_runner.py
   Adheres to api.structure.working.rule.md Section 3.8.
================================================================================
"""

import os
import json
import sqlite3
from typing import Optional, Any, Dict, List
from pathlib import Path




class DatabaseMigrationRunner:
    def __init__(self, db_url: Optional[str] = None) -> None:
        self.db_url = db_url or os.environ.get("DATABASE_URL", "sqlite:///:memory:")
        self.migrations_dir = Path(__file__).resolve().parents[4] / "database" / "migrations"
        self.seeds_file = Path(__file__).resolve().parents[4] / "database" / "seeds" / "algorithm_catalog.json"

    def is_postgres(self) -> bool:
        return self.db_url.startswith("postgres://") or self.db_url.startswith("postgresql://")

    def run_migrations(self, conn: Optional[Any] = None) -> Dict[str, Any]:
        if self.is_postgres():
            return self._run_postgres_migrations(conn)
        else:
            return self._run_sqlite_migrations(conn)

    def rollback_migrations(self, conn: Optional[Any] = None) -> Dict[str, Any]:
        if self.is_postgres():
            return self._rollback_postgres_migrations(conn)
        else:
            return self._rollback_sqlite_migrations(conn)

    def run_graph_migrations(self, graph_port: Optional[Any] = None) -> Dict[str, Any]:
        try:
            from src.domain.ports.graph_port import GraphStorePort
            from src.features.folder_scaffold.queries.folder_scaffold_graph_queries import (
                FLOW_GRAPH_SCHEMA_MIGRATION_STATEMENTS,
            )
            adapter = graph_port
            if adapter is None:
                from src.infra.config.config_loader import AppConfigLoader
                from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
                cfg = AppConfigLoader.get_knowledge_graph_config()
                adapter = Neo4jGraphAdapter(
                    uri=cfg.get("uri"),
                    user=cfg.get("user"),
                    password=cfg.get("password"),
                    database=cfg.get("database", "neo4j"),
                )
            for stmt in FLOW_GRAPH_SCHEMA_MIGRATION_STATEMENTS:
                adapter.query_cypher(stmt)
            return {
                "status": "success",
                "applied_statements_count": len(FLOW_GRAPH_SCHEMA_MIGRATION_STATEMENTS),
                "db": "neo4j",
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "db": "neo4j"}

    def rollback_graph_migrations(self, graph_port: Optional[Any] = None) -> Dict[str, Any]:
        try:
            from src.domain.ports.graph_port import GraphStorePort
            from src.features.folder_scaffold.queries.folder_scaffold_graph_queries import (
                FLOW_GRAPH_SCHEMA_ROLLBACK_STATEMENTS,
            )
            adapter = graph_port
            if adapter is None:
                from src.infra.config.config_loader import AppConfigLoader
                from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
                cfg = AppConfigLoader.get_knowledge_graph_config()
                adapter = Neo4jGraphAdapter(
                    uri=cfg.get("uri"),
                    user=cfg.get("user"),
                    password=cfg.get("password"),
                    database=cfg.get("database", "neo4j"),
                )
            for stmt in FLOW_GRAPH_SCHEMA_ROLLBACK_STATEMENTS:
                adapter.query_cypher(stmt)
            return {
                "status": "success",
                "rolled_back_statements_count": len(FLOW_GRAPH_SCHEMA_ROLLBACK_STATEMENTS),
                "db": "neo4j",
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc), "db": "neo4j"}

    def run_all_migrations(self) -> Dict[str, Any]:
        sql_res = self.run_migrations()
        graph_res = self.run_graph_migrations()
        return {
            "sql": sql_res,
            "graph": graph_res,
            "status": "success" if sql_res.get("status") == "success" and graph_res.get("status") != "error" else "partial_or_error",
        }

    def _run_sqlite_migrations(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        close_conn = False
        if conn is None:
            db_path = self.db_url.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            close_conn = True

        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()

        cursor.execute("SELECT version FROM schema_migrations")
        applied_versions = {row[0] for row in cursor.fetchall()}

        sqlite_files = sorted(
            [f for f in self.migrations_dir.glob("*_sqlite.sql") if not f.name.endswith(".rollback.sql")]
        )

        applied = []
        for file in sqlite_files:
            version = file.stem.split("_")[0]
            if version in applied_versions:
                continue
            sql = file.read_text(encoding="utf-8")
            conn.executescript(sql)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, name) VALUES (?, ?)",
                (version, file.stem),
            )
            conn.commit()
            applied.append(version)

        if close_conn:
            conn.close()

        return {"status": "success", "applied_versions": applied, "db": "sqlite"}

    def _rollback_sqlite_migrations(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        close_conn = False
        if conn is None:
            db_path = self.db_url.replace("sqlite:///", "")
            conn = sqlite3.connect(db_path)
            close_conn = True

        rollback_files = sorted(
            [f for f in self.migrations_dir.glob("*_sqlite.rollback.sql")],
            reverse=True,
        )

        rolled_back = []
        for file in rollback_files:
            version = file.name.split("_")[0]
            sql = file.read_text(encoding="utf-8")
            conn.executescript(sql)
            conn.commit()
            rolled_back.append(version)

        if close_conn:
            conn.close()

        return {"status": "success", "rolled_back_versions": rolled_back, "db": "sqlite"}

    def _run_postgres_migrations(self, conn: Optional[Any] = None) -> Dict[str, Any]:
        import psycopg2

        close_conn = False
        if conn is None:
            conn = psycopg2.connect(self.db_url)
            close_conn = True

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version VARCHAR(64) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("SELECT version FROM schema_migrations")
            applied_versions = {row[0] for row in cur.fetchall()}

        pg_files = sorted(
            [
                f for f in self.migrations_dir.glob("*.sql")
                if not f.name.endswith(".rollback.sql") and "_sqlite" not in f.name
            ]
        )

        applied = []
        with conn.cursor() as cur:
            for file in pg_files:
                version = file.stem.split("_")[0]
                if version in applied_versions:
                    continue
                sql = file.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute(
                    """
                    INSERT INTO schema_migrations (version, name)
                    VALUES (%s, %s)
                    ON CONFLICT (version) DO NOTHING
                    """,
                    (version, file.stem),
                )
                applied.append(version)
        conn.commit()

        if close_conn:
            conn.close()

        return {"status": "success", "applied_versions": applied, "db": "postgres"}

    def _rollback_postgres_migrations(self, conn: Optional[Any] = None) -> Dict[str, Any]:
        import psycopg2

        close_conn = False
        if conn is None:
            conn = psycopg2.connect(self.db_url)
            close_conn = True

        rollback_files = sorted(
            [
                f for f in self.migrations_dir.glob("*.rollback.sql")
                if "_sqlite" not in f.name
            ],
            reverse=True,
        )

        rolled_back = []
        with conn.cursor() as cur:
            for file in rollback_files:
                version = file.name.split("_")[0]
                sql = file.read_text(encoding="utf-8")
                cur.execute(sql)
                rolled_back.append(version)
        conn.commit()

        if close_conn:
            conn.close()

        return {"status": "success", "rolled_back_versions": rolled_back, "db": "postgres"}



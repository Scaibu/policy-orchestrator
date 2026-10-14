"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 DEPENDENCIES & SERVICE LIFECYCLE
================================================================================

Provides singleton service initialization, dependency injection accessors,
and storage/engine factory instantiation for the Policy Orchestrator REST API.
"""

from __future__ import annotations
import os
from typing import Dict, Any

from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.infra.adapters.tools.in_memory_tool_registry_adapter import InMemoryToolRegistryAdapter
from src.infra.adapters.agent.in_memory_agent_registry_adapter import InMemoryAgentManifestRegistryAdapter
from src.infra.adapters.database import (
    DatabaseMigrationRunner,
)


def get_orchestrator_services() -> Dict[str, Any]:
    graph_backend = os.environ.get("GRAPH_BACKEND", "inmemory")
    
    if graph_backend == "neo4j":
        graph_store = Neo4jGraphAdapter(
            uri=os.environ.get("NEO4J_URI", "http://localhost:7474"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
    else:
        graph_store = InMemoryGraphAdapter()

    tool_registry = InMemoryToolRegistryAdapter()
    agent_registry = InMemoryAgentManifestRegistryAdapter(load_builtins=True)

    db_url = os.environ.get("DATABASE_URL")
    auto_migrate = os.environ.get("AUTO_MIGRATE", "true").lower() == "true"

    if auto_migrate and db_url:
        try:
            migration_runner = DatabaseMigrationRunner(db_url)
            migration_runner.run_migrations()
        except Exception:
            pass

    return {
        "graph_store": graph_store,
        "agent_registry": agent_registry,
        "tool_registry": tool_registry,
    }


_SERVICES = None


def get_services() -> Dict[str, Any]:
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = get_orchestrator_services()
    return _SERVICES


def get_graph_store() -> Any:
    return get_services().get("graph_store")

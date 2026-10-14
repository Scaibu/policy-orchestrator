"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: REST API V1 DEPENDENCIES & SERVICE LIFECYCLE
================================================================================

Provides singleton service initialization, dependency injection accessors,
and storage/engine factory instantiation for the Policy Orchestrator REST API.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional

RAGService = None
AuditService = None
AgentService = None
from src.infra.adapters.knowledge.policy_rules_loader import PolicyRulesMarkdownLoader
from src.infra.adapters.vector.in_memory_vector_adapter import InMemoryCosineVectorAdapter
from src.infra.adapters.vector.qdrant_vector_adapter import QdrantVectorAdapter
from src.infra.adapters.graph.in_memory_graph_adapter import InMemoryGraphAdapter
from src.infra.adapters.graph.neo4j_adapter import Neo4jGraphAdapter
from src.infra.adapters.search.duckduckgo_search_adapter import DuckDuckGoSearchAdapter
from src.infra.adapters.search.mock_search_adapter import MockWebSearchAdapter
from src.infra.adapters.tools.in_memory_tool_registry_adapter import InMemoryToolRegistryAdapter
from src.infra.adapters.agent.in_memory_agent_registry_adapter import InMemoryAgentManifestRegistryAdapter
from src.infra.adapters.llm.openai_compatible_adapter import OpenAICompatibleAdapter
from src.infra.adapters.llm.mock_llm_adapter import MockLLMAdapter
from src.infra.adapters.database import (
    DatabaseMigrationRunner,
)


def get_orchestrator_services() -> Dict[str, Any]:
    rules_dir = os.environ.get("POLICY_RULES_DIR", "../rules")
    llm_backend = os.environ.get("LLM_BACKEND", "mock")
    vector_backend = os.environ.get("VECTOR_BACKEND", "inmemory")
    graph_backend = os.environ.get("GRAPH_BACKEND", "inmemory")
    search_backend = os.environ.get("SEARCH_BACKEND", "mock")
    
    knowledge_source = PolicyRulesMarkdownLoader(base_rules_dir=rules_dir)
    
    if vector_backend == "qdrant":
        vector_store = QdrantVectorAdapter(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.environ.get("QDRANT_COLLECTION", "policy_rules"),
            vector_size=int(os.environ.get("VECTOR_SIZE", "64")),
        )
    else:
        vector_store = InMemoryCosineVectorAdapter()
    
    if graph_backend == "neo4j":
        graph_store = Neo4jGraphAdapter(
            uri=os.environ.get("NEO4J_URI", "http://localhost:7474"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
    else:
        graph_store = InMemoryGraphAdapter()

    if search_backend == "duckduckgo":
        search_provider = DuckDuckGoSearchAdapter()
    else:
        search_provider = MockWebSearchAdapter()

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


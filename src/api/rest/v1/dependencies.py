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
    InMemoryAlgorithmRegistryAdapter,
    SQLiteAlgorithmRegistryAdapter,
    AlloyDBAlgorithmRegistryAdapter,
    PostgresAlgorithmRegistryAdapter,
    DatabaseMigrationRunner,
)
from src.features.code_engine.service.algorithm_composer_service import AlgorithmComposerService
from src.features.code_engine.service.code_engine_service import CodeEngineService


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

    if db_url and (db_url.startswith("postgres://") or db_url.startswith("postgresql://")):
        if auto_migrate:
            try:
                migration_runner = DatabaseMigrationRunner(db_url)
                migration_runner.run_migrations()
                migration_runner.seed_algorithm_catalog()
            except Exception:
                pass
        algo_registry = AlloyDBAlgorithmRegistryAdapter(db_url)
    elif db_url and db_url.startswith("sqlite://"):
        if auto_migrate:
            try:
                migration_runner = DatabaseMigrationRunner(db_url)
                migration_runner.run_migrations()
                migration_runner.seed_algorithm_catalog()
            except Exception:
                pass
        db_path = db_url.replace("sqlite:///", "")
        algo_registry = SQLiteAlgorithmRegistryAdapter(db_path)
    else:
        algo_registry = SQLiteAlgorithmRegistryAdapter(":memory:")
        if auto_migrate:
            try:
                migration_runner = DatabaseMigrationRunner("sqlite:///:memory:")
                migration_runner.run_migrations(conn=algo_registry._memory_conn)
                migration_runner.seed_algorithm_catalog(conn=algo_registry._memory_conn)
            except Exception:
                pass

    composer_svc = AlgorithmComposerService(algo_registry)
    code_engine_svc = CodeEngineService()

    if llm_backend == "openai":
        llm_provider = OpenAICompatibleAdapter(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model_name=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )
    elif llm_backend == "ollama":
        llm_provider = OpenAICompatibleAdapter(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="EMPTY",
            model_name=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        )
    else:
        llm_provider = MockLLMAdapter()

    rag_svc = (
        RAGService(
            knowledge_source=knowledge_source,
            vector_store=vector_store,
            llm_provider=llm_provider,
        )
        if RAGService is not None
        else None
    )
    audit_svc = AuditService() if AuditService is not None else None
    agent_svc = (
        AgentService(
            llm_provider=llm_provider,
            rag_service=rag_svc,
            audit_service=audit_svc,
            search_provider=search_provider,
            tool_registry=tool_registry,
        )
        if AgentService is not None
        else None
    )
    from src.features.code_engine.service.algorithm_registry_service import AlgorithmRegistryService
    algo_registry_svc = AlgorithmRegistryService(algo_registry)

    return {
        "graph_store": graph_store,
        "agent_registry": agent_registry,
        "tool_registry": tool_registry,
        "algo_registry": algo_registry,
        "algo_registry_service": algo_registry_svc,
        "composer": composer_svc,
        "code_engine": code_engine_svc,
    }


_SERVICES = None


def get_services() -> Dict[str, Any]:
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = get_orchestrator_services()
    return _SERVICES


def get_code_engine_service() -> CodeEngineService:
    return get_services()["code_engine"]


def get_algorithm_registry_service() -> Any:
    return get_services()["algo_registry_service"]


def get_graph_store() -> Any:
    return get_services().get("graph_store")


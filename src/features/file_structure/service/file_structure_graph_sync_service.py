"""
================================================================================
ALGORITHM & ARCHITECTURE BLUEPRINT: COMPREHENSIVE GRAPH SYNC SERVICE
================================================================================

1. OVERVIEW & OBJECTIVE:
   Single Responsibility: Establishes all comprehensive architectural nodes
   and directed dependency relationships in the Knowledge Graph across all
   system tiers:
   - Contracts: OpenAPI, GraphQL, gRPC, AsyncAPI, JSON-Schema
   - Ingress Delivery: REST Router/Handler, GraphQL Resolver/Loader, gRPC Handler, Event Consumer
   - Domain Core: Context Boundary, Domain Types, Schema ACL, Domain Model, Domain Service
   - Decision & Behavioral: RuleSets, State Machines, Workflows, CQRS Read Models
   - Persistence: Repository Ports/Adapters, Named Queries, Migrations, Schema Lock, Database Pool
   - Messaging: Event Producers, Kafka Topics, Topics Lock, Schema Registry, Dead Letter Queues
   - Cross-Feature: Pub/Sub choreographies, Shared Domain Ports
   - Shared & Infra: Shared Libraries, Config Schema, OTEL Tracers, Test Suites

2. ARCHITECTURAL LAYOUT & DESIGN PILLARS:
   - Single Responsibility Principle (SRP): Only owns graph synchronization and edge linking.
   - Zero-Inline-Comment Doctrine: All edge invariants are documented in this header.
================================================================================
"""

from typing import Dict, Any, Optional, Set, List
from pathlib import Path
from src.domain.ports.graph_port import (
    GraphStorePort,
    GraphNode,
    GraphRelationship,
)
from src.infra.filesystem.file_walker import stream_repository_files

class FileStructureGraphSyncService:
    def __init__(self, graph_store: GraphStorePort) -> None:
        self.graph_store = graph_store

    def register_node(
        self,
        node_id: str,
        label: str,
        file_path: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        props = properties or {}
        props["file_path"] = file_path
        props["node_id"] = node_id
        props["label"] = label
        node = GraphNode(id=node_id, label=label, properties=props)
        self.graph_store.upsert_node(node)

    def register_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        rel = GraphRelationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=relationship_type,
            properties=properties or {},
        )
        self.graph_store.upsert_relationship(rel)

    def sync_feature_dag(
        self,
        feature_name: str,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
        port_type: str = "local",
        package_root: Optional[str] = None,
        owner: Optional[str] = None,
        status: str = "active",
        flags: Optional[str] = None,
        migrations: Optional[str] = None,
        contract_version: str = "v1",
    ) -> None:
        f = feature_name.lower().replace("-", "_")

        # 1. Feature-Local Domain Roles (src/features/{f}/)
        context_id = f"context_{f}"
        index_id = f"index_{f}"
        types_id = f"types_{f}"
        acl_id = f"acl_{f}_schema"
        query_id = f"query_{f}_sql"
        port_id = f"port_{f}_repo" if port_type == "local" else "port_shared_impact_analysis"
        adapter_id = f"adapter_{f}_repo" if port_type == "local" else "adapter_shared_neo4j"
        rules_id = f"rules_{f}"
        machine_id = f"machine_{f}"
        workflow_id = f"workflow_{f}"
        service_id = f"service_{f}"

        # 2. External Package Subsystem: Contracts (contracts/)
        contract_openapi_id = f"contract_openapi_{f}"
        contract_graphql_id = f"contract_graphql_{f}"
        contract_grpc_id = f"contract_grpc_{f}"
        contract_asyncapi_id = f"contract_asyncapi_{f}"

        # 3. External Package Subsystem: Ingress Delivery (src/api/)
        router_id = f"router_{f}"
        handler_id = f"handler_{f}"
        route_rules_id = f"route_rules_{f}"
        resolver_id = f"resolver_{f}"
        grpc_handler_id = f"grpc_handler_{f}"
        consumer_id = f"consumer_{f}"

        # 4. External Package Subsystem: Database Persistence & Infra (database/ & src/infra/database/)
        migration_id = f"migration_{f}_0001"
        schema_lock_id = "schema_lock_db"
        db_pool_id = "infra_db_pool"

        # 5. External Package Subsystem: Messaging & Infra (messaging/ & src/infra/messaging/)
        producer_id = f"producer_{f}_events"
        topic_id = f"topic_prod_{f}_events_v1"
        topics_lock_id = "topics_lock_messaging"
        dlq_id = f"dlq_{f}_events"

        # 6. External Package Subsystem: Shared, Config, Observability (src/shared/, config/, src/infra/)
        shared_lib_id = "shared_utils_common"
        config_schema_id = "config_env_schema"
        tracer_id = "infra_otel_tracer"
        test_suite_id = f"test_suite_{f}"

        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""
        p_root = f"{package_root.rstrip('/')}/" if package_root else ""

        # 1. Feature-Local Domain Roles
        context_id = f"{p_prefix}context_{f}"
        index_id = f"{p_prefix}index_{f}"
        types_id = f"{p_prefix}types_{f}"
        acl_id = f"{p_prefix}acl_{f}_schema"
        query_id = f"{p_prefix}query_{f}_sql"
        port_id = f"{p_prefix}port_{f}_repo" if port_type == "local" else f"{p_prefix}port_shared_impact_analysis"
        adapter_id = f"{p_prefix}adapter_{f}_repo" if port_type == "local" else f"{p_prefix}adapter_shared_neo4j"
        rules_id = f"{p_prefix}rules_{f}"
        machine_id = f"{p_prefix}machine_{f}"
        workflow_id = f"{p_prefix}workflow_{f}"
        service_id = f"{p_prefix}service_{f}"

        # 2. Package Subsystem: Contracts
        contract_openapi_id = f"{p_prefix}contract_openapi_{f}"
        contract_graphql_id = f"{p_prefix}contract_graphql_{f}"
        contract_grpc_id = f"{p_prefix}contract_grpc_{f}"
        contract_asyncapi_id = f"{p_prefix}contract_asyncapi_{f}"

        # 3. Package Subsystem: Ingress Delivery
        router_id = f"{p_prefix}router_{f}"
        handler_id = f"{p_prefix}handler_{f}"
        route_rules_id = f"{p_prefix}route_rules_{f}"
        resolver_id = f"{p_prefix}resolver_{f}"
        grpc_handler_id = f"{p_prefix}grpc_handler_{f}"
        consumer_id = f"{p_prefix}consumer_{f}"

        # 4. Package Subsystem: Database Persistence & Infra (Package-Local)
        migration_id = f"{p_prefix}migration_{f}_0001"
        schema_lock_id = f"{p_prefix}schema_lock_db"
        db_pool_id = f"{p_prefix}infra_db_pool"

        # 5. Package Subsystem: Messaging & Infra (Package-Local)
        producer_id = f"{p_prefix}producer_{f}_events"
        topic_id = f"{p_prefix}topic_prod_{f}_events_v1"
        topics_lock_id = f"{p_prefix}topics_lock_messaging"
        dlq_id = f"{p_prefix}dlq_{f}_events"

        # 6. Package Subsystem: Shared, Config, Observability (Package-Local Intra-Feature Sharing)
        shared_lib_id = f"{p_prefix}shared_utils_common"
        config_schema_id = f"{p_prefix}config_env_schema"
        tracer_id = f"{p_prefix}infra_otel_tracer"
        test_suite_id = f"{p_prefix}test_suite_{f}"

        common_props = {
            "feature_name": f,
            "package_name": pkg_slug or "root",
            "status": status,
            "owner": owner or f"@{f}-team",
            "contract": f"{p_root}contracts/openapi/{f}.{contract_version}.yaml",
            "contract_version": contract_version,
            "flags": flags or f"{f}.enabled",
            "migrations": migrations or f"0001_create_{f}_table",
        }

        # Register Feature-Local Nodes
        self.register_node(context_id, "ContextBoundary", f"{p_root}src/features/{f}/context.yml", {**common_props, "layer": "DomainCore"})
        self.register_node(index_id, "PublicFacade", f"{p_root}src/features/{f}/index.py", {**common_props, "layer": "DomainCore"})
        self.register_node(types_id, "DomainTypes", f"{p_root}src/features/{f}/types/{f}_types.py", {**common_props, "layer": "DomainCore"})
        self.register_node(acl_id, "SchemaACL", f"{p_root}src/features/{f}/schema/{f}_schema.py", {**common_props, "layer": "DomainCore"})
        self.register_node(query_id, "NamedQuery", f"{p_root}src/features/{f}/queries/{f}.queries.sql", {**common_props, "layer": "Persistence"})
        
        if port_type == "local":
            self.register_node(port_id, "RepositoryPort", f"{p_root}src/features/{f}/repository/{f}_repository_port.py", {**common_props, "layer": "Persistence"})
            self.register_node(adapter_id, "RepositoryAdapter", f"{p_root}src/features/{f}/repository/{f}_repository.py", {**common_props, "layer": "Persistence"})
        else:
            self.register_node(port_id, "SharedDomainPort", f"{p_root}src/domain/ports/impact_analysis_port.py", {**common_props, "layer": "Persistence"})
            self.register_node(adapter_id, "SharedInfraAdapter", f"{p_root}src/infra/adapters/graph/neo4j_adapter.py", {**common_props, "layer": "Persistence"})

        self.register_node(rules_id, "RuleSet", f"{p_root}src/features/{f}/rules/{f}_rules.py", {**common_props, "layer": "Behavioral"})
        self.register_node(machine_id, "StateMachine", f"{p_root}src/features/{f}/machines/{f}_machine.py", {**common_props, "layer": "Behavioral"})
        self.register_node(workflow_id, "WorkflowEngine", f"{p_root}src/features/{f}/workflows/{f}_workflow.py", {**common_props, "layer": "Behavioral"})
        self.register_node(service_id, "DomainService", f"{p_root}src/features/{f}/service/{f}_service.py", {**common_props, "layer": "DomainCore"})

        # Register External Package Nodes
        self.register_node(contract_openapi_id, "Contract", f"{p_root}contracts/openapi/{f}.v1.yaml", {**common_props, "layer": "Contracts"})
        self.register_node(contract_graphql_id, "ContractGraphQL", f"{p_root}contracts/graphql/{f}.graphql", {**common_props, "layer": "Contracts"})
        self.register_node(contract_grpc_id, "ContractGRPC", f"{p_root}contracts/proto/v1/{f}.proto", {**common_props, "layer": "Contracts"})
        self.register_node(contract_asyncapi_id, "ContractAsyncAPI", f"{p_root}contracts/asyncapi/{f}.v1.yaml", {**common_props, "layer": "Contracts"})

        if with_router:
            self.register_node(router_id, "IngressRouter", f"{p_root}src/api/rest/v1/routers/{f}_router.py", {**common_props, "layer": "IngressDelivery"})
        if with_handler:
            self.register_node(handler_id, "IngressHandler", f"{p_root}src/api/rest/v1/handlers/{f}_handler.py", {**common_props, "layer": "IngressDelivery"})
        if with_route_rules:
            self.register_node(route_rules_id, "RouteRules", f"{p_root}src/api/rest/v1/route.rules/{f}_route_rules.py", {**common_props, "layer": "IngressDelivery"})
        
        self.register_node(resolver_id, "GraphQLResolver", f"{p_root}src/api/graphql/v1/resolvers/{f}_resolver.py", {**common_props, "layer": "IngressDelivery"})
        self.register_node(grpc_handler_id, "GRPCHandler", f"{p_root}src/api/grpc/v1/handlers/{f}_grpc_handler.py", {**common_props, "layer": "IngressDelivery"})
        self.register_node(consumer_id, "EventConsumer", f"{p_root}src/api/events/consumers/{f}_event_consumer.py", {**common_props, "layer": "IngressDelivery"})

        self.register_node(migration_id, "DatabaseMigration", f"{p_root}database/migrations/0001_create_{f}_table.sql", {**common_props, "layer": "Persistence"})
        self.register_node(schema_lock_id, "SchemaLock", f"{p_root}database/schema.lock", {**common_props, "layer": "Persistence"})
        self.register_node(db_pool_id, "DatabasePool", f"{p_root}src/infra/database/pool/connection_pool.py", {**common_props, "layer": "Persistence"})

        self.register_node(producer_id, "EventProducer", f"{p_root}src/infra/messaging/producers/{f}_event_producer.py", {**common_props, "layer": "Messaging"})
        self.register_node(topic_id, "TopicSpec", f"{p_root}messaging/topics/0001_create_{f}_events.json", {**common_props, "layer": "Messaging"})
        self.register_node(topics_lock_id, "TopicsLock", f"{p_root}messaging/topics.lock", {**common_props, "layer": "Messaging"})
        self.register_node(dlq_id, "DeadLetterQueue", f"{p_root}messaging/dlq/{f}_dlq_policy.yaml", {**common_props, "layer": "Messaging"})

        # Intra-Package Shared Infrastructure (Shared within package, isolated across packages)
        self.register_node(shared_lib_id, "SharedLibrary", f"{p_root}src/shared/utils/formatters.py", {**common_props, "layer": "SharedInfrastructure"})
        self.register_node(config_schema_id, "EnvConfigSchema", f"{p_root}config/env.schema", {**common_props, "layer": "SharedInfrastructure"})
        self.register_node(tracer_id, "OpenTelemetryTracer", f"{p_root}src/infra/observability/tracing/tracer.py", {**common_props, "layer": "SharedInfrastructure"})
        self.register_node(test_suite_id, "TestSuite", f"{p_root}src/features/{f}/tests/{f}_rules_test.py", {**common_props, "layer": "DomainCore"})

        # Establish Intra-Feature and Hexagonal Cross-Boundary Edges
        self.register_edge(context_id, service_id, "GOVERNS_BOUNDARY")
        self.register_edge(index_id, service_id, "EXPORTS_FACADE")
        self.register_edge(types_id, acl_id, "TYPED_BY")

        # Ingress Delivery Edges
        if with_router:
            self.register_edge(contract_openapi_id, router_id, "DEFINES_ROUTE")
            if with_route_rules:
                self.register_edge(router_id, route_rules_id, "EVALUATES_ROUTE_RULES")
            if with_handler:
                self.register_edge(router_id, handler_id, "DELEGATES_TO")

        if with_handler:
            self.register_edge(handler_id, acl_id, "INVOKES_ACL")
            self.register_edge(handler_id, service_id, "INVOKES_DOMAIN")

        self.register_edge(contract_graphql_id, resolver_id, "DEFINES_GRAPHQL")
        self.register_edge(resolver_id, service_id, "RESOLVES_TO")
        self.register_edge(contract_grpc_id, grpc_handler_id, "DEFINES_GRPC")
        self.register_edge(grpc_handler_id, service_id, "DISPATCHES_GRPC")
        self.register_edge(contract_asyncapi_id, topic_id, "SPECIFIES_ASYNC")

        # Domain Logic Edges
        self.register_edge(service_id, rules_id, "ENFORCES_RULES")
        self.register_edge(service_id, machine_id, "TRANSITIONS_STATE")
        self.register_edge(service_id, workflow_id, "RUNS_WORKFLOW")

        # Persistence Edges
        self.register_edge(service_id, port_id, "CONSUMES_PORT")
        self.register_edge(port_id, adapter_id, "IMPLEMENTED_BY")
        self.register_edge(adapter_id, query_id, "EXECUTES_QUERY")
        self.register_edge(adapter_id, db_pool_id, "ACQUIRES_CONNECTION")
        self.register_edge(query_id, migration_id, "DEPENDS_ON_DDL")
        self.register_edge(migration_id, schema_lock_id, "LOCKED_BY")

        # Messaging Edges
        self.register_edge(service_id, producer_id, "EMITS_EVENT")
        self.register_edge(producer_id, topic_id, "PUBLISHES_TO")
        self.register_edge(topic_id, topics_lock_id, "LOCKED_BY")
        self.register_edge(topic_id, consumer_id, "DELIVERS_TO")
        self.register_edge(topic_id, dlq_id, "FALLBACK_DLQ")
        self.register_edge(consumer_id, service_id, "TRIGGERS_DOMAIN")

        # Intra-Package Shared Infrastructure Edges
        self.register_edge(service_id, shared_lib_id, "IMPORTS_SHARED")
        self.register_edge(service_id, config_schema_id, "CONFIGURED_BY")
        self.register_edge(service_id, tracer_id, "TRACED_BY")
        self.register_edge(test_suite_id, service_id, "TESTS_COMPONENT")

        if package_root:
            pkg_node_id = f"pkg_{pkg_slug}"
            self.register_edge(pkg_node_id, service_id, "CONTAINS_FEATURE")

    def sync_package_dag(self, package_name: str, target_root: str) -> None:
        pkg_slug = package_name.lower().replace("-", "_")
        pkg_node_id = f"pkg_{pkg_slug}"
        self.register_node(
            node_id=pkg_node_id,
            label="PackageRoot",
            file_path=target_root,
            properties={"name": pkg_slug},
        )

    def link_database_migration(self, feature_name: str, migration_file: str) -> None:
        f = feature_name.lower().replace("-", "_")
        migration_id = f"migration_{Path(migration_file).stem}"
        query_id = f"query_{f}_sql"
        self.register_node(migration_id, "DatabaseMigration", migration_file)
        self.register_edge(query_id, migration_id, "DEPENDS_ON_DDL")

    def link_kafka_event_topic(
        self,
        feature_name: str,
        event_name: str,
        topic_name: str,
        topic_spec_path: Optional[str] = None,
    ) -> None:
        f = feature_name.lower().replace("-", "_")
        service_id = f"service_{f}"
        producer_id = f"producer_{f}_{event_name.lower()}"
        topic_id = f"topic_{topic_name.replace('.', '_').replace('-', '_')}"

        self.register_node(producer_id, "EventProducer", f"src/features/{f}/repository/{f}.event.producer.py")
        self.register_node(topic_id, "TopicSpec", topic_spec_path or f"messaging/topics/{topic_name}.yaml")

        self.register_edge(service_id, producer_id, "EMITS_EVENT")
        self.register_edge(producer_id, topic_id, "PUBLISHES_TO")

    def link_cross_feature_event(
        self,
        publisher_feature: str,
        consumer_feature: str,
        event_name: str,
        topic_name: str,
    ) -> None:
        pub_f = publisher_feature.lower().replace("-", "_")
        sub_f = consumer_feature.lower().replace("-", "_")
        topic_id = f"topic_{topic_name.replace('.', '_').replace('-', '_')}"
        consumer_id = f"consumer_{sub_f}_{event_name.lower()}"
        sub_service_id = f"service_{sub_f}"

        self.link_kafka_event_topic(publisher_feature, event_name, topic_name)
        self.register_node(consumer_id, "EventConsumer", f"src/api/events/consumers/{sub_f}_{event_name.lower()}_consumer.py")

        self.register_edge(topic_id, consumer_id, "DELIVERS_TO")
        self.register_edge(consumer_id, sub_service_id, "TRIGGERS_DOMAIN")

    def link_shared_dependency(self, feature_name: str, shared_file: str) -> None:
        f = feature_name.lower().replace("-", "_")
        service_id = f"service_{f}"
        shared_id = f"shared_{Path(shared_file).stem}"
        self.register_node(shared_id, "SharedLibrary", shared_file)
        self.register_edge(service_id, shared_id, "IMPORTS_SHARED")

    def scan_and_index_repository(
        self,
        root_dir: str,
        allowed_extensions: Optional[Set[str]] = None,
        auto_sync_features: bool = True,
    ) -> Dict[str, Any]:
        exts = allowed_extensions or {".py", ".sql", ".yaml", ".yml", ".json", ".graphql", ".proto", ".sh", ".md"}
        indexed_count = 0
        root_path = Path(root_dir).resolve()
        discovered_features: Set[str] = set()
        feature_packages: Dict[str, Optional[str]] = {}

        for fpath in stream_repository_files(root_dir=root_dir, allowed_extensions=exts):
            try:
                rel_path = str(fpath.relative_to(root_path))
            except ValueError:
                rel_path = str(fpath)
            node_id = f"file_{rel_path.replace('/', '_').replace('.', '_')}"
            label = self._infer_label_from_path(rel_path)

            self.register_node(
                node_id=node_id,
                label=label,
                file_path=rel_path,
                properties={
                    "absolute_path": str(fpath),
                    "size_bytes": fpath.stat().st_size if fpath.exists() else 0,
                    "extension": fpath.suffix,
                }
            )
            indexed_count += 1

            # Auto-detect features from file path
            parts = Path(rel_path).parts
            if "features" in parts:
                idx = parts.index("features")
                if idx + 1 < len(parts):
                    feat_slug = parts[idx + 1].lower().replace("-", "_")
                    discovered_features.add(feat_slug)
                    if "packages" in parts:
                        pkg_idx = parts.index("packages")
                        if pkg_idx + 1 < idx:
                            feature_packages[feat_slug] = "/".join(parts[:pkg_idx + 2])
                    else:
                        feature_packages.setdefault(feat_slug, None)

        # Automatically establish spatial and hexagonal DAG relationships for all discovered features
        synced_feature_count = 0
        if auto_sync_features:
            for feat in discovered_features:
                pkg_root = feature_packages.get(feat)
                if pkg_root:
                    self.sync_package_dag(package_name=Path(pkg_root).name, target_root=pkg_root)
                self.sync_feature_dag(
                    feature_name=feat,
                    package_root=pkg_root,
                )
                synced_feature_count += 1

        return {
            "indexed_files": indexed_count,
            "discovered_features": len(discovered_features),
            "features": sorted(list(discovered_features)),
            "synced_features_count": synced_feature_count,
            "status": "COMPLETED",
        }

    def register_and_link_file(
        self,
        file_path: str,
        role: str,
        rel_type: str,
        target_file_or_node: str,
        direction: str = "outgoing",
        package_root: Optional[str] = None,
        feature_name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not file_path:
            raise ValueError("file_path must not be empty")
        if not role:
            raise ValueError("role/label must not be empty")
        if not rel_type:
            raise ValueError("rel_type is a required parameter when creating/registering a file")
        if not target_file_or_node:
            raise ValueError("target_file_or_node is a required parameter to link the new file")

        p_root = f"{package_root.rstrip('/')}/" if package_root else ""
        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""

        f_slug = feature_name.lower().replace("-", "_") if feature_name else ""
        if not f_slug:
            p_parts = Path(file_path).parts
            if "features" in p_parts:
                idx = p_parts.index("features")
                if idx + 1 < len(p_parts):
                    f_slug = p_parts[idx + 1].lower().replace("-", "_")

        stem = Path(file_path).stem.lower().replace("-", "_")
        node_id = f"{p_prefix}{role.lower()}_{stem}"

        props = properties or {}
        if f_slug:
            props["feature_name"] = f_slug
        if pkg_slug:
            props["package_name"] = pkg_slug
        if "layer" not in props:
            props["layer"] = self._infer_layer_from_role(role)

        self.register_node(node_id=node_id, label=role, file_path=file_path, properties=props)

        resolved_target_id = self._resolve_target_node_id(target_file_or_node, package_root=package_root, feature_name=f_slug)

        if direction.lower() == "incoming":
            src_id, tgt_id = resolved_target_id, node_id
        else:
            src_id, tgt_id = node_id, resolved_target_id

        self.register_edge(source_id=src_id, target_id=tgt_id, relationship_type=rel_type)

        return {
            "node_id": node_id,
            "label": role,
            "file_path": file_path,
            "relationship": {
                "source_id": src_id,
                "target_id": tgt_id,
                "rel_type": rel_type,
                "direction": direction,
            }
        }

    def link_files(
        self,
        source_file_or_node: str,
        rel_type: str,
        target_file_or_node: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not source_file_or_node or not target_file_or_node or not rel_type:
            raise ValueError("source, target, and rel_type are required to link files")

        src_id = self._resolve_target_node_id(source_file_or_node)
        tgt_id = self._resolve_target_node_id(target_file_or_node)

        self.register_edge(source_id=src_id, target_id=tgt_id, relationship_type=rel_type, properties=properties)

        return {
            "source_id": src_id,
            "target_id": tgt_id,
            "relationship_type": rel_type,
            "linked": True,
        }

    def _resolve_target_node_id(self, target_query: str, package_root: Optional[str] = None, feature_name: Optional[str] = None) -> str:
        q = target_query.strip()
        node = self.graph_store.find_node(q)
        if node:
            return node.id

        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""
        return f"{p_prefix}{q.replace('/', '_').replace('.', '_')}"


    def _infer_layer_from_role(self, role: str) -> str:
        r = role.lower()
        if r in ["domainservice", "schemacl", "domaintypes", "contextboundary", "publicfacade"]:
            return "DomainCore"
        if r in ["ingressrouter", "ingresshandler", "routerules", "graphqlresolver", "grpchandler", "eventconsumer"]:
            return "IngressDelivery"
        if r in ["ruleset", "statemachine", "workflowengine"]:
            return "Behavioral"
        if r in ["repositoryport", "repositoryadapter", "namedquery", "databasemigration", "schemalock", "databasepool"]:
            return "Persistence"
        if r in ["contract", "contractgraphql", "contractgrpc", "contractasyncapi"]:
            return "Contracts"
        if r in ["eventproducer", "topicspec", "topicslock", "deadletterqueue"]:
            return "Messaging"
        if r in ["sharedlibrary", "envconfigschema", "opentelemetrytracer", "testsuite"]:
            return "SharedInfrastructure"
        return "DomainCore"

    def _infer_label_from_path(self, path: str) -> str:
        p = path.lower()
        if "contracts/openapi" in p or p.endswith(".yaml") or p.endswith(".json"):
            return "Contract"
        if "router" in p and "routers/" in p:
            return "IngressRouter"
        if "handler" in p:
            return "IngressHandler"
        if "schema" in p:
            return "SchemaACL"
        if "service" in p:
            return "DomainService"
        if "rule" in p:
            return "RuleSet"
        if "machine" in p:
            return "StateMachine"
        if "workflow" in p:
            return "WorkflowEngine"
        if "port" in p:
            return "RepositoryPort"
        if "adapter" in p or "repository" in p:
            return "RepositoryAdapter"
        if "queries" in p or p.endswith(".sql"):
            return "NamedQuery"
        if "migration" in p:
            return "DatabaseMigration"
        return "Unknown"

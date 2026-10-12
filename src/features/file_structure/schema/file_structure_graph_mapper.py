from pathlib import Path
from typing import Optional, List, Tuple
from src.features.file_structure.types.file_structure_types import (
    FeatureGraphNodeIdsModel,
    FeatureGraphContextModel,
    FeatureGraphNodeSpec,
)

class FeatureGraphMapper:
    @staticmethod
    def map_to_graph_context(
        feature_name: str,
        port_type: str = "local",
        package_root: Optional[str] = None,
        owner: Optional[str] = None,
        status: str = "active",
        flags: Optional[str] = None,
        migrations: Optional[str] = None,
        contract_version: str = "v1",
    ) -> FeatureGraphContextModel:
        f = feature_name.lower().replace("-", "_")
        pkg_slug = Path(package_root).name.lower().replace("-", "_") if package_root else ""
        p_prefix = f"pkg_{pkg_slug}_" if pkg_slug else ""
        p_root = f"{package_root.rstrip('/')}/" if package_root else ""

        node_ids = FeatureGraphNodeIdsModel(
            context_id=f"{p_prefix}context_{f}",
            index_id=f"{p_prefix}index_{f}",
            types_id=f"{p_prefix}types_{f}",
            acl_id=f"{p_prefix}acl_{f}_schema",
            query_id=f"{p_prefix}query_{f}_sql",
            port_id=f"{p_prefix}port_{f}_repo" if port_type == "local" else f"{p_prefix}port_shared_impact_analysis",
            adapter_id=f"{p_prefix}adapter_{f}_repo" if port_type == "local" else f"{p_prefix}adapter_shared_neo4j",
            rules_id=f"{p_prefix}rules_{f}",
            machine_id=f"{p_prefix}machine_{f}",
            workflow_id=f"{p_prefix}workflow_{f}",
            service_id=f"{p_prefix}service_{f}",
            contract_openapi_id=f"{p_prefix}contract_openapi_{f}",
            contract_graphql_id=f"{p_prefix}contract_graphql_{f}",
            contract_grpc_id=f"{p_prefix}contract_grpc_{f}",
            contract_asyncapi_id=f"{p_prefix}contract_asyncapi_{f}",
            router_id=f"{p_prefix}router_{f}",
            handler_id=f"{p_prefix}handler_{f}",
            route_rules_id=f"{p_prefix}route_rules_{f}",
            resolver_id=f"{p_prefix}resolver_{f}",
            grpc_handler_id=f"{p_prefix}grpc_handler_{f}",
            consumer_id=f"{p_prefix}consumer_{f}",
            migration_id=f"{p_prefix}migration_{f}_0001",
            schema_lock_id=f"{p_prefix}schema_lock_db",
            db_pool_id=f"{p_prefix}infra_db_pool",
            producer_id=f"{p_prefix}producer_{f}_events",
            topic_id=f"{p_prefix}topic_prod_{f}_events_v1",
            topics_lock_id=f"{p_prefix}topics_lock_messaging",
            dlq_id=f"{p_prefix}dlq_{f}_events",
            shared_lib_id=f"{p_prefix}shared_utils_common",
            config_schema_id=f"{p_prefix}config_env_schema",
            tracer_id=f"{p_prefix}infra_otel_tracer",
            test_suite_id=f"{p_prefix}test_suite_{f}",
        )

        return FeatureGraphContextModel(
            feature_name=f,
            package_name=pkg_slug or "root",
            package_root=p_root,
            package_prefix=p_prefix,
            status=status,
            owner=owner or f"@{f}-team",
            contract=f"{p_root}contracts/openapi/{f}.{contract_version}.yaml",
            contract_version=contract_version,
            flags=flags or f"{f}.enabled",
            migrations=migrations or f"0001_create_{f}_table",
            node_ids=node_ids,
        )

    @staticmethod
    def map_to_node_specs(
        ctx: FeatureGraphContextModel,
        port_type: str = "local",
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
    ) -> List[FeatureGraphNodeSpec]:
        f = ctx.feature_name
        p_root = ctx.package_root
        ids = ctx.node_ids

        specs = [
            FeatureGraphNodeSpec(ids.context_id, "ContextBoundary", f"{p_root}src/features/{f}/context.yml", "DomainCore"),
            FeatureGraphNodeSpec(ids.index_id, "PublicFacade", f"{p_root}src/features/{f}/index.py", "DomainCore"),
            FeatureGraphNodeSpec(ids.types_id, "DomainTypes", f"{p_root}src/features/{f}/types/{f}_types.py", "DomainCore"),
            FeatureGraphNodeSpec(ids.acl_id, "SchemaACL", f"{p_root}src/features/{f}/schema/{f}_schema.py", "DomainCore"),
            FeatureGraphNodeSpec(ids.query_id, "NamedQuery", f"{p_root}src/features/{f}/queries/{f}.queries.sql", "Persistence"),
            FeatureGraphNodeSpec(ids.rules_id, "RuleSet", f"{p_root}src/features/{f}/rules/{f}_rules.py", "Behavioral"),
            FeatureGraphNodeSpec(ids.machine_id, "StateMachine", f"{p_root}src/features/{f}/machines/{f}_machine.py", "Behavioral"),
            FeatureGraphNodeSpec(ids.workflow_id, "WorkflowEngine", f"{p_root}src/features/{f}/workflows/{f}_workflow.py", "Behavioral"),
            FeatureGraphNodeSpec(ids.service_id, "DomainService", f"{p_root}src/features/{f}/service/{f}_service.py", "DomainCore"),
            FeatureGraphNodeSpec(ids.contract_openapi_id, "Contract", f"{p_root}contracts/openapi/{f}.v1.yaml", "Contracts"),
            FeatureGraphNodeSpec(ids.contract_graphql_id, "ContractGraphQL", f"{p_root}contracts/graphql/{f}.graphql", "Contracts"),
            FeatureGraphNodeSpec(ids.contract_grpc_id, "ContractGRPC", f"{p_root}contracts/proto/v1/{f}.proto", "Contracts"),
            FeatureGraphNodeSpec(ids.contract_asyncapi_id, "ContractAsyncAPI", f"{p_root}contracts/asyncapi/{f}.v1.yaml", "Contracts"),
            FeatureGraphNodeSpec(ids.resolver_id, "GraphQLResolver", f"{p_root}src/api/graphql/v1/resolvers/{f}_resolver.py", "IngressDelivery"),
            FeatureGraphNodeSpec(ids.grpc_handler_id, "GRPCHandler", f"{p_root}src/api/grpc/v1/handlers/{f}_grpc_handler.py", "IngressDelivery"),
            FeatureGraphNodeSpec(ids.consumer_id, "EventConsumer", f"{p_root}src/api/events/consumers/{f}_event_consumer.py", "IngressDelivery"),
            FeatureGraphNodeSpec(ids.migration_id, "DatabaseMigration", f"{p_root}database/migrations/0001_create_{f}_table.sql", "Persistence"),
            FeatureGraphNodeSpec(ids.schema_lock_id, "SchemaLock", f"{p_root}database/schema.lock", "Persistence"),
            FeatureGraphNodeSpec(ids.db_pool_id, "DatabasePool", f"{p_root}src/infra/database/pool/connection_pool.py", "Persistence"),
            FeatureGraphNodeSpec(ids.producer_id, "EventProducer", f"{p_root}src/infra/messaging/producers/{f}_event_producer.py", "Messaging"),
            FeatureGraphNodeSpec(ids.topic_id, "TopicSpec", f"{p_root}messaging/topics/0001_create_{f}_events.json", "Messaging"),
            FeatureGraphNodeSpec(ids.topics_lock_id, "TopicsLock", f"{p_root}messaging/topics.lock", "Messaging"),
            FeatureGraphNodeSpec(ids.dlq_id, "DeadLetterQueue", f"{p_root}messaging/dlq/{f}_dlq_policy.yaml", "Messaging"),
            FeatureGraphNodeSpec(ids.shared_lib_id, "SharedLibrary", f"{p_root}src/shared/utils/formatters.py", "SharedInfrastructure"),
            FeatureGraphNodeSpec(ids.config_schema_id, "EnvConfigSchema", f"{p_root}config/env.schema", "SharedInfrastructure"),
            FeatureGraphNodeSpec(ids.tracer_id, "OpenTelemetryTracer", f"{p_root}src/infra/observability/tracing/tracer.py", "SharedInfrastructure"),
            FeatureGraphNodeSpec(ids.test_suite_id, "TestSuite", f"{p_root}src/features/{f}/tests/{f}_rules_test.py", "DomainCore"),
        ]

        if port_type == "local":
            specs.append(FeatureGraphNodeSpec(ids.port_id, "RepositoryPort", f"{p_root}src/features/{f}/repository/{f}_repository_port.py", "Persistence"))
            specs.append(FeatureGraphNodeSpec(ids.adapter_id, "RepositoryAdapter", f"{p_root}src/features/{f}/repository/{f}_repository.py", "Persistence"))
        else:
            specs.append(FeatureGraphNodeSpec(ids.port_id, "SharedDomainPort", f"{p_root}src/domain/ports/impact_analysis_port.py", "Persistence"))
            specs.append(FeatureGraphNodeSpec(ids.adapter_id, "SharedInfraAdapter", f"{p_root}src/infra/adapters/graph/neo4j_adapter.py", "Persistence"))

        if with_router:
            specs.append(FeatureGraphNodeSpec(ids.router_id, "IngressRouter", f"{p_root}src/api/rest/v1/routers/{f}_router.py", "IngressDelivery"))
        if with_handler:
            specs.append(FeatureGraphNodeSpec(ids.handler_id, "IngressHandler", f"{p_root}src/api/rest/v1/handlers/{f}_handler.py", "IngressDelivery"))
        if with_route_rules:
            specs.append(FeatureGraphNodeSpec(ids.route_rules_id, "RouteRules", f"{p_root}src/api/rest/v1/route.rules/{f}_route_rules.py", "IngressDelivery"))

        return specs

    @staticmethod
    def map_to_edge_specs(
        ctx: FeatureGraphContextModel,
        with_router: bool = True,
        with_handler: bool = True,
        with_route_rules: bool = False,
    ) -> List[Tuple[str, str, str]]:
        ids = ctx.node_ids
        edges = [
            (ids.context_id, ids.service_id, "GOVERNS_BOUNDARY"),
            (ids.index_id, ids.service_id, "EXPORTS_FACADE"),
            (ids.types_id, ids.acl_id, "TYPED_BY"),
            (ids.contract_graphql_id, ids.resolver_id, "DEFINES_GRAPHQL"),
            (ids.resolver_id, ids.service_id, "RESOLVES_TO"),
            (ids.contract_grpc_id, ids.grpc_handler_id, "DEFINES_GRPC"),
            (ids.grpc_handler_id, ids.service_id, "DISPATCHES_GRPC"),
            (ids.contract_asyncapi_id, ids.topic_id, "SPECIFIES_ASYNC"),
            (ids.service_id, ids.rules_id, "ENFORCES_RULES"),
            (ids.service_id, ids.machine_id, "TRANSITIONS_STATE"),
            (ids.service_id, ids.workflow_id, "RUNS_WORKFLOW"),
            (ids.service_id, ids.port_id, "CONSUMES_PORT"),
            (ids.port_id, ids.adapter_id, "IMPLEMENTED_BY"),
            (ids.adapter_id, ids.query_id, "EXECUTES_QUERY"),
            (ids.adapter_id, ids.db_pool_id, "ACQUIRES_CONNECTION"),
            (ids.query_id, ids.migration_id, "DEPENDS_ON_DDL"),
            (ids.migration_id, ids.schema_lock_id, "LOCKED_BY"),
            (ids.service_id, ids.producer_id, "EMITS_EVENT"),
            (ids.producer_id, ids.topic_id, "PUBLISHES_TO"),
            (ids.topic_id, ids.topics_lock_id, "LOCKED_BY"),
            (ids.topic_id, ids.consumer_id, "DELIVERS_TO"),
            (ids.topic_id, ids.dlq_id, "FALLBACK_DLQ"),
            (ids.consumer_id, ids.service_id, "TRIGGERS_DOMAIN"),
            (ids.service_id, ids.shared_lib_id, "IMPORTS_SHARED"),
            (ids.service_id, ids.config_schema_id, "CONFIGURED_BY"),
            (ids.service_id, ids.tracer_id, "TRACED_BY"),
            (ids.test_suite_id, ids.service_id, "TESTS_COMPONENT"),
        ]

        if with_router:
            edges.append((ids.contract_openapi_id, ids.router_id, "DEFINES_ROUTE"))
            if with_route_rules:
                edges.append((ids.router_id, ids.route_rules_id, "EVALUATES_ROUTE_RULES"))
            if with_handler:
                edges.append((ids.router_id, ids.handler_id, "DELEGATES_TO"))

        if with_handler:
            edges.append((ids.handler_id, ids.acl_id, "INVOKES_ACL"))
            edges.append((ids.handler_id, ids.service_id, "INVOKES_DOMAIN"))

        if ctx.package_root:
            pkg_slug = Path(ctx.package_root).name.lower().replace("-", "_")
            edges.append((f"pkg_{pkg_slug}", ids.service_id, "CONTAINS_FEATURE"))

        return edges

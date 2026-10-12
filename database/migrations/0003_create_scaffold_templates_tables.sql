-- migration:      0003
-- description:    create scaffold templates, features, and packages tables
-- author:         Antigravity / Lead Software Engineer
-- date:           2026-10-12
-- depends_on:     0001
-- reversible:     YES
-- lock_risk:      LOW
-- rows_affected:  2 (schema creation & initial seed)
-- reason:         Dynamic database-driven scaffolding templates and audit tracking

CREATE TABLE IF NOT EXISTS scaffold_templates (
    id VARCHAR(128) PRIMARY KEY,
    template_type VARCHAR(64) NOT NULL,
    version VARCHAR(32) NOT NULL DEFAULT 'v1',
    directories JSONB NOT NULL,
    files JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT scaffold_templates_type_ck CHECK (template_type IN ('FEATURE', 'PACKAGE')),
    CONSTRAINT scaffold_templates_type_version_uk UNIQUE (template_type, version),
    CONSTRAINT scaffold_templates_dirs_arr_ck CHECK (jsonb_typeof(directories) = 'array'),
    CONSTRAINT scaffold_templates_files_arr_ck CHECK (jsonb_typeof(files) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_scaffold_templates_type ON scaffold_templates (template_type);
CREATE INDEX IF NOT EXISTS idx_scaffold_templates_active ON scaffold_templates (is_active);

CREATE TABLE IF NOT EXISTS scaffolded_features (
    feature_name VARCHAR(128) PRIMARY KEY,
    target_dir TEXT NOT NULL,
    port_type VARCHAR(32) NOT NULL DEFAULT 'local',
    files JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT scaffolded_features_port_type_ck CHECK (port_type IN ('local', 'shared')),
    CONSTRAINT scaffolded_features_files_arr_ck CHECK (jsonb_typeof(files) = 'array')
);

CREATE TABLE IF NOT EXISTS scaffolded_packages (
    package_name VARCHAR(128) PRIMARY KEY,
    target_root TEXT NOT NULL,
    directories JSONB NOT NULL,
    files JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT scaffolded_packages_dirs_arr_ck CHECK (jsonb_typeof(directories) = 'array'),
    CONSTRAINT scaffolded_packages_files_arr_ck CHECK (jsonb_typeof(files) = 'array')
);

INSERT INTO scaffold_templates (
    id,
    template_type,
    version,
    directories,
    files,
    metadata,
    is_active,
    updated_at
) VALUES 
(
    'tpl_feature_canonical',
    'FEATURE',
    'v1',
    '["schema", "types", "service", "rules", "machines", "workflows", "repository", "queries", "tests"]'::jsonb,
    '["context.yml", "index.py", "schema/{feature_slug}_schema.py", "types/{feature_slug}_types.py", "service/{feature_slug}_service.py", "rules/{feature_slug}_rules.py", "machines/{feature_slug}_machine.py", "workflows/{feature_slug}_workflow.py", "repository/{feature_slug}_repository_port.py", "repository/{feature_slug}_repository.py", "queries/{feature_slug}.queries.sql", "tests/{feature_slug}_rules_test.py", "tests/{feature_slug}_machine_test.py"]'::jsonb,
    '{"architecture": "Hexagonal Domain Core"}'::jsonb,
    TRUE,
    CURRENT_TIMESTAMP
),
(
    'tpl_package_canonical',
    'PACKAGE',
    'v1',
    '["contracts/openapi", "contracts/graphql", "contracts/proto/v1", "contracts/asyncapi", "contracts/json-schema", "config", "database/migrations", "database/rls", "database/indexes", "database/partitioning", "database/storage_engine", "database/replication", "database/consensus", "database/quorums", "database/cdc", "database/sharding", "database/crdts", "database/anti_entropy", "database/fencing", "database/retention", "database/seeds", "messaging/topics", "messaging/schema-registry", "messaging/dlq", "messaging/subscriptions", "deploy/k8s", "deploy/helm", "src/api/rest/v1/router", "src/api/rest/v1/routers", "src/api/rest/v1/handlers", "src/api/graphql/v1/schema", "src/api/graphql/v1/resolvers", "src/api/graphql/v1/dataloaders", "src/api/grpc/v1/server", "src/api/grpc/v1/handlers", "src/api/events/consumers", "src/api/events/publishers", "src/features", "src/infra/config", "src/infra/database/pool", "src/infra/database/factory", "src/infra/database/transaction", "src/infra/database/executor", "src/infra/database/middleware", "src/infra/database/migrations", "src/infra/database/tracing", "src/infra/database/adapters", "src/infra/messaging/broker", "src/infra/messaging/factory", "src/infra/messaging/producers", "src/infra/messaging/consumers", "src/infra/messaging/middleware", "src/infra/messaging/topics", "src/infra/messaging/migrations", "src/infra/messaging/tracing", "src/infra/messaging/cqrs", "src/infra/clients", "src/infra/observability/tracing", "src/infra/observability/clocks", "src/infra/observability/profiling", "src/infra/observability/race_detection", "src/infra/observability/deadlock", "src/infra/observability/heap_analysis", "src/infra/observability/ebpf", "src/infra/observability/wire_analysis", "src/infra/observability/vector_inspection", "src/infra/observability/divergence_audit", "src/infra/observability/idempotency_audit", "src/infra/observability/transition_log", "src/infra/observability/snapshots", "src/infra/observability/replay", "src/infra/observability/wal_miner", "src/infra/observability/shadow_traffic", "src/infra/observability/circuit_breaker_history", "src/infra/observability/saturation_analysis", "src/infra/observability/analytics", "src/infra/observability/topology", "src/shared/utils", "src/shared/constants", "src/shared/errors", "src/shared/types", "scripts"]'::jsonb,
    '["contracts/openapi/v1.yaml", "contracts/openapi/changelog.md", "contracts/graphql/v1.graphql", "contracts/graphql/changelog.md", "contracts/asyncapi/v1.yaml", "contracts/changelog.md", "config/env.schema", "config/default.yaml", "config/development.yaml", "config/production.yaml", "config/test.yaml", "config/feature-flags.yaml", "database/migrations/0001_initial_schema.sql", "database/migrations/0001_initial_schema.rollback.sql", "database/rls/0001_tenant_isolation_rls.sql", "database/indexes/0001_performance_indexes.sql", "database/partitioning/range_partitioning.yaml", "database/partitioning/hash_partitioning.yaml", "database/partitioning/directory_partitioning.json", "database/storage_engine/lsm_storage.yaml", "database/storage_engine/tiered_storage.yaml", "database/storage_engine/hot_cold_archiving.sql", "database/storage_engine/polyglot_dispatch.json", "database/replication/topology_spec.yaml", "database/replication/chain_replication.yaml", "database/consensus/raft_consensus.yaml", "database/consensus/wal_shipping.yaml", "database/quorums/quorum_config.yaml", "database/quorums/ack_policy.yaml", "database/cdc/cdc_publisher_spec.json", "database/cdc/cdc_sink_spec.json", "database/sharding/shard_hash_ring.yaml", "database/crdts/crdt_definitions.yaml", "database/crdts/hybrid_clock.yaml", "database/anti_entropy/merkle_tree_sync.sql", "database/fencing/fencing_epoch_failover.sql", "database/retention/soft_delete_30day_purge.sql", "database/seeds/dev.seed.sql", "database/seeds/test.seed.sql", "database/schema.lock", "messaging/topics/0001_create_user_events.json", "messaging/topics/0001_create_user_events.rollback.json", "messaging/schema-registry/user_events.v1.json", "messaging/dlq/dlq_policy.yaml", "messaging/subscriptions/consumer_groups.yaml", "messaging/topics.lock", "deploy/k8s/deployment.yaml", "deploy/k8s/service.yaml", "deploy/k8s/hpa.yaml", "deploy/k8s/configmap.yaml", "deploy/helm/Chart.yaml", "deploy/helm/values.yaml", "scripts/run.sh", "scripts/migrate.sh", "scripts/test.sh", "scripts/generate.sh", "Dockerfile", "Dockerfile.dev", "docker-compose.yml", ".dockerignore", ".env.example", ".package-meta.yaml", ".port-registry"]'::jsonb,
    '{"architecture": "Polyglot Distributed Package Infrastructure"}'::jsonb,
    TRUE,
    CURRENT_TIMESTAMP
)
ON CONFLICT (id) DO UPDATE SET
    directories = EXCLUDED.directories,
    files = EXCLUDED.files,
    metadata = EXCLUDED.metadata,
    updated_at = CURRENT_TIMESTAMP;

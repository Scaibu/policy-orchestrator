-- ================================================================================
-- ALGORITHM & ARCHITECTURE BLUEPRINT: FILE STRUCTURE NAMED SQL QUERIES
-- ================================================================================
-- Formula: FLOW_{VERB}_{ENTITY}_{CRITERIA}
-- All queries are parameterized. Raw inline SQL construction is strictly forbidden.
-- ================================================================================

-- name: FLOW_INSERT_FILE_NODE
INSERT INTO file_structure_nodes (
    id,
    label,
    file_path,
    feature_name,
    properties_json,
    created_at,
    updated_at
) VALUES (
    :id,
    :label,
    :file_path,
    :feature_name,
    :properties_json,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
)
ON CONFLICT (id) DO UPDATE SET
    label = EXCLUDED.label,
    file_path = EXCLUDED.file_path,
    feature_name = EXCLUDED.feature_name,
    properties_json = EXCLUDED.properties_json,
    updated_at = CURRENT_TIMESTAMP;

-- name: FLOW_GET_FILE_NODE_BY_ID
SELECT
    id,
    label,
    file_path,
    feature_name,
    properties_json,
    created_at,
    updated_at
FROM file_structure_nodes
WHERE id = :id;

-- name: FLOW_FIND_FILE_NODES_BY_FEATURE_NAME
SELECT
    id,
    label,
    file_path,
    feature_name,
    properties_json,
    created_at,
    updated_at
FROM file_structure_nodes
WHERE feature_name = :feature_name
ORDER BY label ASC, id ASC;

-- name: FLOW_FIND_FILE_NODES_BY_LABEL
SELECT
    id,
    label,
    file_path,
    feature_name,
    properties_json,
    created_at,
    updated_at
FROM file_structure_nodes
WHERE label = :label
ORDER BY feature_name ASC, id ASC;

-- name: FLOW_LIST_ALL_FILE_NODES
SELECT
    id,
    label,
    file_path,
    feature_name,
    properties_json,
    created_at,
    updated_at
FROM file_structure_nodes
ORDER BY feature_name ASC, label ASC, id ASC;

-- name: FLOW_COUNT_FILE_NODES_BY_FEATURE
SELECT
    feature_name,
    COUNT(id) AS total_nodes
FROM file_structure_nodes
GROUP BY feature_name
ORDER BY total_nodes DESC;

-- name: FLOW_INSERT_DEPENDENCY_EDGE
INSERT INTO file_structure_edges (
    source_id,
    target_id,
    relationship_type,
    properties_json,
    created_at
) VALUES (
    :source_id,
    :target_id,
    :relationship_type,
    :properties_json,
    CURRENT_TIMESTAMP
)
ON CONFLICT (source_id, target_id, relationship_type) DO UPDATE SET
    properties_json = EXCLUDED.properties_json;

-- name: FLOW_GET_DEPENDENCY_EDGE_BY_ENDPOINTS
SELECT
    source_id,
    target_id,
    relationship_type,
    properties_json,
    created_at
FROM file_structure_edges
WHERE source_id = :source_id
  AND target_id = :target_id
  AND relationship_type = :relationship_type;

-- name: FLOW_LIST_UPSTREAM_DEPENDENCIES_BY_TARGET_ID
SELECT
    e.source_id AS id,
    n.label,
    n.file_path,
    n.feature_name,
    e.relationship_type AS relationship,
    e.properties_json AS edge_properties
FROM file_structure_edges e
JOIN file_structure_nodes n ON e.source_id = n.id
WHERE e.target_id = :target_id
ORDER BY n.label ASC, n.id ASC;

-- name: FLOW_LIST_DOWNSTREAM_DEPENDENTS_BY_SOURCE_ID
SELECT
    e.target_id AS id,
    n.label,
    n.file_path,
    n.feature_name,
    e.relationship_type AS relationship,
    e.properties_json AS edge_properties
FROM file_structure_edges e
JOIN file_structure_nodes n ON e.target_id = n.id
WHERE e.source_id = :source_id
ORDER BY n.label ASC, n.id ASC;

-- name: FLOW_LIST_ALL_EDGES_BY_RELATIONSHIP_TYPE
SELECT
    e.source_id,
    sn.label AS source_label,
    e.target_id,
    tn.label AS target_label,
    e.relationship_type,
    e.properties_json,
    e.created_at
FROM file_structure_edges e
JOIN file_structure_nodes sn ON e.source_id = sn.id
JOIN file_structure_nodes tn ON e.target_id = tn.id
WHERE e.relationship_type = :relationship_type
ORDER BY e.source_id ASC, e.target_id ASC;

-- name: FLOW_COUNT_EDGES_BY_TYPE
SELECT
    relationship_type,
    COUNT(1) AS total_edges
FROM file_structure_edges
GROUP BY relationship_type
ORDER BY total_edges DESC;

-- name: FLOW_DELETE_FILE_NODE_BY_ID
DELETE FROM file_structure_nodes
WHERE id = :id;

-- name: FLOW_DELETE_EDGES_BY_NODE_ID
DELETE FROM file_structure_edges
WHERE source_id = :id OR target_id = :id;

-- name: FLOW_DELETE_ALL_GRAPH_RECORDS
DELETE FROM file_structure_edges;
DELETE FROM file_structure_nodes;

-- name: FLOW_FIND_ORPHAN_NODES
SELECT
    n.id,
    n.label,
    n.file_path,
    n.feature_name
FROM file_structure_nodes n
LEFT JOIN file_structure_edges es ON n.id = es.source_id
LEFT JOIN file_structure_edges et ON n.id = et.target_id
WHERE es.source_id IS NULL AND et.target_id IS NULL
ORDER BY n.feature_name ASC, n.label ASC;

-- name: FLOW_LIST_FEATURE_TOPOLOGY_SUMMARY
SELECT
    n.feature_name,
    COUNT(DISTINCT n.id) AS total_nodes,
    COUNT(DISTINCT e.source_id || '-' || e.target_id) AS total_edges
FROM file_structure_nodes n
LEFT JOIN file_structure_edges e ON n.id = e.source_id
GROUP BY n.feature_name
ORDER BY n.feature_name ASC;

-- name: FLOW_GET_SCAFFOLD_TEMPLATE_BY_TYPE_AND_VERSION
SELECT
    id,
    template_type,
    version,
    directories,
    files,
    metadata,
    is_active,
    created_at,
    updated_at
FROM scaffold_templates
WHERE template_type = :template_type
  AND version = :version
  AND is_active = TRUE;

-- name: FLOW_INSERT_SCAFFOLD_TEMPLATE
INSERT INTO scaffold_templates (
    id,
    template_type,
    version,
    directories,
    files,
    metadata,
    is_active,
    created_at,
    updated_at
) VALUES (
    :id,
    :template_type,
    :version,
    :directories,
    :files,
    :metadata,
    :is_active,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
)
ON CONFLICT (id) DO UPDATE SET
    directories = EXCLUDED.directories,
    files = EXCLUDED.files,
    metadata = EXCLUDED.metadata,
    is_active = EXCLUDED.is_active,
    updated_at = CURRENT_TIMESTAMP;

-- name: FLOW_LIST_SCAFFOLD_TEMPLATES
SELECT
    id,
    template_type,
    version,
    directories,
    files,
    metadata,
    is_active,
    created_at,
    updated_at
FROM scaffold_templates
ORDER BY template_type ASC, version ASC;

-- name: FLOW_INSERT_SCAFFOLDED_FEATURE
INSERT INTO scaffolded_features (
    feature_name,
    target_dir,
    port_type,
    files,
    metadata,
    created_at
) VALUES (
    :feature_name,
    :target_dir,
    :port_type,
    :files,
    :metadata,
    CURRENT_TIMESTAMP
)
ON CONFLICT (feature_name) DO UPDATE SET
    target_dir = EXCLUDED.target_dir,
    port_type = EXCLUDED.port_type,
    files = EXCLUDED.files,
    metadata = EXCLUDED.metadata;

-- name: FLOW_GET_SCAFFOLDED_FEATURE_BY_NAME
SELECT
    feature_name,
    target_dir,
    port_type,
    files,
    metadata,
    created_at
FROM scaffolded_features
WHERE feature_name = :feature_name;

-- name: FLOW_INSERT_SCAFFOLDED_PACKAGE
INSERT INTO scaffolded_packages (
    package_name,
    target_root,
    directories,
    files,
    metadata,
    created_at
) VALUES (
    :package_name,
    :target_root,
    :directories,
    :files,
    :metadata,
    CURRENT_TIMESTAMP
)
ON CONFLICT (package_name) DO UPDATE SET
    target_root = EXCLUDED.target_root,
    directories = EXCLUDED.directories,
    files = EXCLUDED.files,
    metadata = EXCLUDED.metadata;

-- name: FLOW_GET_SCAFFOLDED_PACKAGE_BY_NAME
SELECT
    package_name,
    target_root,
    directories,
    files,
    metadata,
    created_at
FROM scaffolded_packages
WHERE package_name = :package_name;

-- name: FLOW_SEED_SCAFFOLD_TEMPLATES
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

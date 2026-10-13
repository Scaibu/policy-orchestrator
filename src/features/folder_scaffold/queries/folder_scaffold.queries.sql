-- name: FLOW_INSERT_SCAFFOLDED_NODE
INSERT INTO scaffolded_nodes (id, file_path, role, layer, feature_name, package_name, file_size_bytes, version, updated_at)
VALUES (%(id)s, %(file_path)s, %(role)s, %(layer)s, %(feature_name)s, %(package_name)s, %(file_size_bytes)s, 1, NOW())
ON CONFLICT (id) DO UPDATE SET
    file_path = EXCLUDED.file_path,
    role = EXCLUDED.role,
    layer = EXCLUDED.layer,
    feature_name = EXCLUDED.feature_name,
    package_name = EXCLUDED.package_name,
    file_size_bytes = EXCLUDED.file_size_bytes,
    version = scaffolded_nodes.version + 1,
    updated_at = NOW();

-- name: FLOW_INSERT_SCAFFOLDED_RELATIONSHIP
INSERT INTO scaffolded_relationships (id, source_id, target_id, rel_type, properties)
VALUES (%(id)s, %(source_id)s, %(target_id)s, %(rel_type)s, %(properties)s)
ON CONFLICT (id) DO UPDATE SET
    properties = EXCLUDED.properties;

-- name: FLOW_GET_SCAFFOLDED_NODE_BY_ID
SELECT id, file_path, role, layer, feature_name, package_name, file_size_bytes, version
FROM scaffolded_nodes
WHERE id = %(id)s;

-- name: FLOW_CHECK_SCAFFOLDED_RELATIONSHIP_ORPHANS
SELECT r.id, r.source_id, r.target_id
FROM scaffolded_relationships r
LEFT JOIN scaffolded_nodes s ON r.source_id = s.id
LEFT JOIN scaffolded_nodes t ON r.target_id = t.id
WHERE s.id IS NULL OR t.id IS NULL;

-- name: FLOW_CHECK_SCAFFOLDED_NODE_SIZE_INVARIANT
SELECT id, file_path, file_size_bytes
FROM scaffolded_nodes
WHERE file_size_bytes < 0;

-- name: FLOW_COUNT_SCAFFOLDED_NODES_BY_PACKAGE
SELECT COUNT(*) AS total_nodes, package_name
FROM scaffolded_nodes
WHERE package_name = %(package_name)s
GROUP BY package_name;

-- name: FLOW_LIST_SCAFFOLDED_NODES_BY_PACKAGE
SELECT id, file_path, role, layer, feature_name, package_name, version
FROM scaffolded_nodes
WHERE package_name = %(package_name)s
ORDER BY created_at;

-- name: FLOW_COUNT_SCAFFOLDED_RELATIONSHIPS_BY_PACKAGE
SELECT COUNT(*) AS total_relationships
FROM scaffolded_relationships sr
JOIN scaffolded_nodes sn ON sr.source_id = sn.id
WHERE sn.package_name = %(package_name)s;

-- name: FLOW_LIST_SCAFFOLDED_RELATIONSHIPS_BY_PACKAGE
SELECT
    sr.rel_type,
    src.file_path AS source_path,
    tgt.file_path AS target_path,
    sr.properties
FROM scaffolded_relationships sr
JOIN scaffolded_nodes src ON sr.source_id = src.id
JOIN scaffolded_nodes tgt ON sr.target_id = tgt.id
WHERE src.package_name = %(package_name)s
ORDER BY sr.rel_type;

-- name: FLOW_VERIFY_SCAFFOLDED_RELATIONSHIP_INTEGRITY_BY_PACKAGE
SELECT sr.id, sr.source_id, sr.target_id
FROM scaffolded_relationships sr
LEFT JOIN scaffolded_nodes s ON sr.source_id = s.id
LEFT JOIN scaffolded_nodes t ON sr.target_id = t.id
JOIN scaffolded_nodes sn ON sr.source_id = sn.id
WHERE (s.id IS NULL OR t.id IS NULL)
  AND sn.package_name = %(package_name)s;

-- name: FLOW_COUNT_SCAFFOLDED_RELATIONSHIPS_BY_TYPE
SELECT sr.rel_type, COUNT(*) AS count
FROM scaffolded_relationships sr
JOIN scaffolded_nodes sn ON sr.source_id = sn.id
WHERE sn.package_name = %(package_name)s
GROUP BY sr.rel_type
ORDER BY count DESC;

-- name: FLOW_COUNT_SCAFFOLDED_NODES_BY_ROLE
SELECT role, COUNT(*) AS count
FROM scaffolded_nodes
WHERE package_name = %(package_name)s
GROUP BY role
ORDER BY count DESC;

-- name: FLOW_AUDIT_RULE0_SUBSYSTEM_COVERAGE
SELECT
    CASE
        WHEN file_path LIKE '%contracts%' THEN '01-contracts'
        WHEN file_path LIKE '%/config/%'  THEN '02-config'
        WHEN file_path LIKE '%database%'  THEN '03-database'
        WHEN file_path LIKE '%messaging%' THEN '04-messaging'
        WHEN file_path LIKE '%deploy%'    THEN '05-deploy'
        WHEN file_path LIKE '%shared%'    THEN '06-src/shared'
        WHEN file_path LIKE '%infra%'     THEN '07-src/infra'
        WHEN file_path LIKE '%features%'  THEN '08-src/features'
        WHEN file_path LIKE '%/api/%'     THEN '09-src/api'
        WHEN file_path LIKE '%scripts%'   THEN '10-scripts'
        ELSE 'other'
    END AS subsystem,
    COUNT(*) AS node_count
FROM scaffolded_nodes
WHERE package_name = %(package_name)s
GROUP BY subsystem
ORDER BY subsystem;

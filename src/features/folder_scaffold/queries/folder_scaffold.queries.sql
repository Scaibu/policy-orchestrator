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

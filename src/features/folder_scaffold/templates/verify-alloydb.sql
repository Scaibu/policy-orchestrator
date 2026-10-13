-- ============================================================
-- AlloyDB Verification Queries
-- Template: 0-rule-package-structure
-- Run after: folder_scaffold_cli.py scaffold-tree
-- ============================================================

-- 1. Count all scaffolded nodes for this package
SELECT COUNT(*) AS total_nodes, package_name
FROM scaffolded_nodes
WHERE package_name = 'PACKAGE_NAME'
GROUP BY package_name;

-- 2. List every node with its role and layer (ordered by creation)
SELECT id, file_path, role, layer, feature_name, package_name, version
FROM scaffolded_nodes
WHERE package_name = 'PACKAGE_NAME'
ORDER BY created_at;

-- 3. Count all relationships for this package
SELECT COUNT(*) AS total_relationships
FROM scaffolded_relationships sr
JOIN scaffolded_nodes sn ON sr.source_id = sn.id
WHERE sn.package_name = 'PACKAGE_NAME';

-- 4. List every relationship with source and target paths
SELECT
    sr.rel_type,
    src.file_path AS source_path,
    tgt.file_path AS target_path,
    sr.properties
FROM scaffolded_relationships sr
JOIN scaffolded_nodes src ON sr.source_id = src.id
JOIN scaffolded_nodes tgt ON sr.target_id = tgt.id
WHERE src.package_name = 'PACKAGE_NAME'
ORDER BY sr.rel_type;

-- 5. Verify no orphan relationships (referential integrity check)
SELECT sr.id, sr.source_id, sr.target_id
FROM scaffolded_relationships sr
LEFT JOIN scaffolded_nodes s ON sr.source_id = s.id
LEFT JOIN scaffolded_nodes t ON sr.target_id = t.id
WHERE s.id IS NULL OR t.id IS NULL;

-- 6. Relationship type distribution for this package
SELECT sr.rel_type, COUNT(*) AS count
FROM scaffolded_relationships sr
JOIN scaffolded_nodes sn ON sr.source_id = sn.id
WHERE sn.package_name = 'PACKAGE_NAME'
GROUP BY sr.rel_type
ORDER BY count DESC;

-- 7. Nodes by architectural role distribution
SELECT role, COUNT(*) AS count
FROM scaffolded_nodes
WHERE package_name = 'PACKAGE_NAME'
GROUP BY role
ORDER BY count DESC;

-- 8. Verify all 10 mandatory Rule 0 subsystem directories exist
SELECT
    CASE
        WHEN file_path LIKE '%contracts%'      THEN '01-contracts'
        WHEN file_path LIKE '%config%'         THEN '02-config'
        WHEN file_path LIKE '%database%'       THEN '03-database'
        WHEN file_path LIKE '%messaging%'      THEN '04-messaging'
        WHEN file_path LIKE '%deploy%'         THEN '05-deploy'
        WHEN file_path LIKE '%shared%'         THEN '06-src/shared'
        WHEN file_path LIKE '%infra%'          THEN '07-src/infra'
        WHEN file_path LIKE '%features%'       THEN '08-src/features'
        WHEN file_path LIKE '%api%'            THEN '09-src/api'
        WHEN file_path LIKE '%scripts%'        THEN '10-scripts'
        ELSE 'other'
    END AS subsystem,
    COUNT(*) AS node_count
FROM scaffolded_nodes
WHERE package_name = 'PACKAGE_NAME'
GROUP BY subsystem
ORDER BY subsystem;

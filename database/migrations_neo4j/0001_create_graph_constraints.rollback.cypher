// rollback:       0001
// description:    drop neo4j constraints and indexes
// author:         Antigravity / Lead Software Engineer
// date:           2026-10-12
// database:       neo4j
// reversible:     YES

DROP CONSTRAINT file_node_id_unique IF EXISTS;
DROP CONSTRAINT feature_id_unique IF EXISTS;
DROP CONSTRAINT package_root_id_unique IF EXISTS;
DROP INDEX file_node_path_idx IF EXISTS;
DROP INDEX file_node_role_idx IF EXISTS;
DROP INDEX file_node_layer_idx IF EXISTS;

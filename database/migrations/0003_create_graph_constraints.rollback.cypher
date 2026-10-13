// rollback:       0003
// description:    drop knowledge graph unique constraints and traversal indexes
// author:         Antigravity / Lead Systems Architect
// date:           2026-10-13

DROP CONSTRAINT file_node_id_unique IF EXISTS;
DROP CONSTRAINT package_root_id_unique IF EXISTS;
DROP CONSTRAINT feature_id_unique IF EXISTS;

DROP INDEX file_node_path_idx IF EXISTS;
DROP INDEX file_node_role_idx IF EXISTS;
DROP INDEX file_node_package_idx IF EXISTS;
DROP INDEX file_node_feature_idx IF EXISTS;
DROP INDEX file_node_layer_idx IF EXISTS;

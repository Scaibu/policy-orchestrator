// migration:      0003
// description:    create knowledge graph unique constraints and traversal indexes
// author:         Antigravity / Lead Systems Architect
// date:           2026-10-13
// reversible:     YES

CREATE CONSTRAINT file_node_id_unique IF NOT EXISTS FOR (n:FileNode) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT package_root_id_unique IF NOT EXISTS FOR (p:PackageRoot) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT feature_id_unique IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE;

CREATE INDEX file_node_path_idx IF NOT EXISTS FOR (n:FileNode) ON (n.file_path);
CREATE INDEX file_node_role_idx IF NOT EXISTS FOR (n:FileNode) ON (n.role);
CREATE INDEX file_node_package_idx IF NOT EXISTS FOR (n:FileNode) ON (n.package_name);
CREATE INDEX file_node_feature_idx IF NOT EXISTS FOR (n:FileNode) ON (n.feature_name);
CREATE INDEX file_node_layer_idx IF NOT EXISTS FOR (n:FileNode) ON (n.layer);

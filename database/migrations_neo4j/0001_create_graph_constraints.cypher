// migration:      0001
// description:    create neo4j constraints and indexes for file nodes and relationships
// author:         Antigravity / Lead Software Engineer
// date:           2026-10-12
// database:       neo4j
// reversible:     YES

CREATE CONSTRAINT file_node_id_unique IF NOT EXISTS
FOR (f:FileNode) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT feature_id_unique IF NOT EXISTS
FOR (feat:Feature) REQUIRE feat.id IS UNIQUE;

CREATE CONSTRAINT package_root_id_unique IF NOT EXISTS
FOR (pkg:PackageRoot) REQUIRE pkg.id IS UNIQUE;

CREATE INDEX file_node_path_idx IF NOT EXISTS
FOR (f:FileNode) ON (f.file_path);

CREATE INDEX file_node_role_idx IF NOT EXISTS
FOR (f:FileNode) ON (f.role);

CREATE INDEX file_node_layer_idx IF NOT EXISTS
FOR (f:FileNode) ON (f.layer);

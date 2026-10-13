// ============================================================
// Neo4j Cypher Verification Queries
// Template: 0-rule-package-structure
// Run after: folder_scaffold_cli.py scaffold-tree --store neo4j
// Replace PACKAGE_NAME with your actual package name
// ============================================================

// 1. Count all FileNode + PackageRoot nodes for this package
MATCH (n)
WHERE n.package_name = 'PACKAGE_NAME' OR n.id = 'PACKAGE_NAME'
RETURN labels(n) AS label, count(*) AS total
ORDER BY total DESC;

// 2. List every node scaffolded under this package
MATCH (n:FileNode)
WHERE n.package_name = 'PACKAGE_NAME'
RETURN n.id AS id, n.file_path AS path, n.role AS role, n.layer AS layer
ORDER BY n.file_path;

// 3. Show the PackageRoot node and all direct children
MATCH (root {id: 'PACKAGE_NAME'})-[r]->(child)
RETURN root.id AS package, type(r) AS rel, child.file_path AS child_path
ORDER BY type(r);

// 4. Count all relationships in the package graph
MATCH (a:FileNode)-[r]->(b:FileNode)
WHERE a.package_name = 'PACKAGE_NAME'
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC;

// 5. Full graph — all nodes and edges for this package
MATCH (a)-[r]->(b)
WHERE a.package_name = 'PACKAGE_NAME' OR a.id = 'PACKAGE_NAME'
RETURN a.file_path AS source, type(r) AS rel, b.file_path AS target
ORDER BY type(r), a.file_path;

// 6. Blast radius: what breaks if contracts/openapi changes? (upstream traversal)
MATCH (target:FileNode {file_path: 'contracts/openapi/.gitkeep'})
      <-[r*1..6]-(impacted)
WHERE impacted.package_name = 'PACKAGE_NAME'
RETURN DISTINCT impacted.file_path AS impacted_node,
       [rel IN r | type(rel)] AS relationship_chain,
       length(r) AS hops
ORDER BY hops, impacted_node;

// 7. File lineage: what does src/features depend on? (downstream traversal)
MATCH (start:FileNode {file_path: 'src/features/.gitkeep'})
      -[r*1..5]->(dependency)
WHERE dependency.package_name = 'PACKAGE_NAME'
RETURN DISTINCT dependency.file_path AS depends_on,
       [rel IN r | type(rel)] AS via,
       length(r) AS depth
ORDER BY depth, depends_on;

// 8. Verify no isolated (orphan) nodes — every node should have at least one relationship
MATCH (n:FileNode)
WHERE n.package_name = 'PACKAGE_NAME'
  AND NOT (n)-[]-()
RETURN n.file_path AS orphan_node, 'WARNING: no relationships' AS status;

// 9. Show the full deployment chain: what leads to deploy/k8s?
MATCH path = (src)-[*1..8]->(deploy:FileNode)
WHERE deploy.file_path CONTAINS 'deploy/k8s'
  AND src.package_name = 'PACKAGE_NAME'
RETURN [node IN nodes(path) | node.file_path] AS full_chain,
       length(path) AS depth
ORDER BY depth DESC
LIMIT 10;

// 10. Relationship type inventory — verify all 32 rel types are present
MATCH (a:FileNode)-[r]->(b:FileNode)
WHERE a.package_name = 'PACKAGE_NAME'
WITH COLLECT(DISTINCT type(r)) AS actual_types
RETURN actual_types,
       size(actual_types) AS distinct_relationship_types;

FLOW_GRAPH_COUNT_NODES_BY_PACKAGE = """
MATCH (n:FileNode {package_name: $package_name})
RETURN count(n) AS total_nodes
"""

FLOW_GRAPH_LIST_NODES_BY_PACKAGE = """
MATCH (n:FileNode {package_name: $package_name})
RETURN n.id AS id, n.file_path AS file_path, n.role AS role, n.layer AS layer
ORDER BY n.file_path
"""

FLOW_GRAPH_COUNT_RELATIONSHIPS_BY_PACKAGE = """
MATCH (a:FileNode {package_name: $package_name})-[r]->(b)
RETURN count(r) AS total_relationships
"""

FLOW_GRAPH_LIST_RELATIONSHIPS_BY_PACKAGE = """
MATCH (a:FileNode {package_name: $package_name})-[r]->(b)
RETURN a.file_path AS source, type(r) AS rel_type, b.file_path AS target
ORDER BY type(r), a.file_path
"""

FLOW_GRAPH_COUNT_RELATIONSHIPS_BY_TYPE = """
MATCH (a:FileNode {package_name: $package_name})-[r]->(b)
RETURN type(r) AS relationship_type, count(*) AS count
ORDER BY count DESC
"""

FLOW_GRAPH_CHECK_ORPHAN_NODES = """
MATCH (n:FileNode {package_name: $package_name})
WHERE NOT (n)-[]-()
RETURN n.file_path AS orphan_node
"""

FLOW_GRAPH_BLAST_RADIUS_UPSTREAM = """
MATCH (target:FileNode {id: $target_id})
      <-[r*1..$max_depth]-(impacted:FileNode)
RETURN DISTINCT impacted.file_path AS impacted_node,
       [rel IN r | type(rel)] AS relationship_chain,
       length(r) AS hops
ORDER BY hops, impacted_node
"""

FLOW_GRAPH_BLAST_RADIUS_DOWNSTREAM = """
MATCH (target:FileNode {id: $target_id})
      -[r*1..$max_depth]->(impacted:FileNode)
RETURN DISTINCT impacted.file_path AS impacted_node,
       [rel IN r | type(rel)] AS relationship_chain,
       length(r) AS hops
ORDER BY hops, impacted_node
"""

FLOW_GRAPH_FILE_LINEAGE_DIRECT_UPSTREAM = """
MATCH (n:FileNode {file_path: $file_path})<-[r]-(upstream)
RETURN upstream.file_path AS dependency, type(r) AS via
ORDER BY type(r)
"""

FLOW_GRAPH_FILE_LINEAGE_DIRECT_DOWNSTREAM = """
MATCH (n:FileNode {file_path: $file_path})-[r]->(downstream)
RETURN downstream.file_path AS dependency, type(r) AS via
ORDER BY type(r)
"""

FLOW_GRAPH_DEPLOYMENT_CHAIN = """
MATCH path = (src:FileNode {package_name: $package_name})-[*1..8]->(deploy:FileNode)
WHERE deploy.file_path CONTAINS 'deploy/k8s'
RETURN [node IN nodes(path) | node.file_path] AS full_chain,
       length(path) AS depth
ORDER BY depth DESC
LIMIT 10
"""

FLOW_GRAPH_AUDIT_DISTINCT_RELATIONSHIP_TYPES = """
MATCH (a:FileNode {package_name: $package_name})-[r]->(b)
WITH collect(DISTINCT type(r)) AS actual_types
RETURN actual_types, size(actual_types) AS distinct_relationship_types
"""

FLOW_GRAPH_CREATE_FILE_NODE_CONSTRAINT = "CREATE CONSTRAINT file_node_id_unique IF NOT EXISTS FOR (n:FileNode) REQUIRE n.id IS UNIQUE"
FLOW_GRAPH_CREATE_PACKAGE_ROOT_CONSTRAINT = "CREATE CONSTRAINT package_root_id_unique IF NOT EXISTS FOR (p:PackageRoot) REQUIRE p.id IS UNIQUE"
FLOW_GRAPH_CREATE_FEATURE_CONSTRAINT = "CREATE CONSTRAINT feature_id_unique IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE"

FLOW_GRAPH_CREATE_FILE_NODE_PATH_INDEX = "CREATE INDEX file_node_path_idx IF NOT EXISTS FOR (n:FileNode) ON (n.file_path)"
FLOW_GRAPH_CREATE_FILE_NODE_ROLE_INDEX = "CREATE INDEX file_node_role_idx IF NOT EXISTS FOR (n:FileNode) ON (n.role)"
FLOW_GRAPH_CREATE_FILE_NODE_PACKAGE_INDEX = "CREATE INDEX file_node_package_idx IF NOT EXISTS FOR (n:FileNode) ON (n.package_name)"
FLOW_GRAPH_CREATE_FILE_NODE_FEATURE_INDEX = "CREATE INDEX file_node_feature_idx IF NOT EXISTS FOR (n:FileNode) ON (n.feature_name)"
FLOW_GRAPH_CREATE_FILE_NODE_LAYER_INDEX = "CREATE INDEX file_node_layer_idx IF NOT EXISTS FOR (n:FileNode) ON (n.layer)"

FLOW_GRAPH_DROP_FILE_NODE_CONSTRAINT = "DROP CONSTRAINT file_node_id_unique IF EXISTS"
FLOW_GRAPH_DROP_PACKAGE_ROOT_CONSTRAINT = "DROP CONSTRAINT package_root_id_unique IF EXISTS"
FLOW_GRAPH_DROP_FEATURE_CONSTRAINT = "DROP CONSTRAINT feature_id_unique IF EXISTS"

FLOW_GRAPH_DROP_FILE_NODE_PATH_INDEX = "DROP INDEX file_node_path_idx IF EXISTS"
FLOW_GRAPH_DROP_FILE_NODE_ROLE_INDEX = "DROP INDEX file_node_role_idx IF EXISTS"
FLOW_GRAPH_DROP_FILE_NODE_PACKAGE_INDEX = "DROP INDEX file_node_package_idx IF EXISTS"
FLOW_GRAPH_DROP_FILE_NODE_FEATURE_INDEX = "DROP INDEX file_node_feature_idx IF EXISTS"
FLOW_GRAPH_DROP_FILE_NODE_LAYER_INDEX = "DROP INDEX file_node_layer_idx IF EXISTS"

FLOW_GRAPH_SCHEMA_MIGRATION_STATEMENTS = (
    FLOW_GRAPH_CREATE_FILE_NODE_CONSTRAINT,
    FLOW_GRAPH_CREATE_PACKAGE_ROOT_CONSTRAINT,
    FLOW_GRAPH_CREATE_FEATURE_CONSTRAINT,
    FLOW_GRAPH_CREATE_FILE_NODE_PATH_INDEX,
    FLOW_GRAPH_CREATE_FILE_NODE_ROLE_INDEX,
    FLOW_GRAPH_CREATE_FILE_NODE_PACKAGE_INDEX,
    FLOW_GRAPH_CREATE_FILE_NODE_FEATURE_INDEX,
    FLOW_GRAPH_CREATE_FILE_NODE_LAYER_INDEX,
)

FLOW_GRAPH_SCHEMA_ROLLBACK_STATEMENTS = (
    FLOW_GRAPH_DROP_FILE_NODE_CONSTRAINT,
    FLOW_GRAPH_DROP_PACKAGE_ROOT_CONSTRAINT,
    FLOW_GRAPH_DROP_FEATURE_CONSTRAINT,
    FLOW_GRAPH_DROP_FILE_NODE_PATH_INDEX,
    FLOW_GRAPH_DROP_FILE_NODE_ROLE_INDEX,
    FLOW_GRAPH_DROP_FILE_NODE_PACKAGE_INDEX,
    FLOW_GRAPH_DROP_FILE_NODE_FEATURE_INDEX,
    FLOW_GRAPH_DROP_FILE_NODE_LAYER_INDEX,
)

FLOW_GRAPH_DELETE_PACKAGE_NODES_AND_RELATIONSHIPS = """
MATCH (n)
WHERE n.package_name = $package_name
   OR (n.file_path IS NOT NULL AND n.file_path CONTAINS $package_name)
   OR (n.id IS NOT NULL AND n.id CONTAINS $package_name)
DETACH DELETE n
"""



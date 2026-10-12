// ================================================================================
// ALGORITHM & ARCHITECTURE BLUEPRINT: FILE STRUCTURE CYPHER QUERIES (NEO4J)
// ================================================================================
//
// SECTION 1: ONE-CLICK BROWSER PARAMS INITIALIZER
// Run this block in the Neo4j Browser (http://localhost:7474) to define all params:
// :params { id: "service_file_structure", feature_name: "file_structure", src: "router_file_structure", tgt: "handler_file_structure", label: "DomainService", file_path: "src/features/file_structure/service/file_structure_service.py", rel_type: "DELEGATES_TO", props: {} }
//
// ================================================================================

// --------------------------------------------------------------------------------
// PART A: READY-TO-RUN BROWSER QUERIES (ZERO PARAMS NEEDED)
// --------------------------------------------------------------------------------

// 1. Interactive Architecture Map for file_structure Feature
MATCH (n)-[r]->(m)
WHERE n.feature_name = "file_structure" OR m.feature_name = "file_structure"
RETURN n, r, m
LIMIT 50;

// 2. Upstream Blast Radius for Service Layer (Who invokes the Service)
MATCH (target:DomainService {id: "service_file_structure"})<-[r]-(caller)
RETURN caller.id AS caller_id,
       labels(caller)[0] AS caller_type,
       caller.file_path AS caller_file_path,
       type(r) AS relationship,
       target.id AS target_id;

// 3. Downstream Dependencies for Service Layer (What the Service invokes)
MATCH (source:DomainService {id: "service_file_structure"})-[r]->(dependency)
RETURN source.id AS source_id,
       type(r) AS relationship,
       dependency.id AS dependency_id,
       labels(dependency)[0] AS dependency_type,
       dependency.file_path AS dependency_file_path;

// 4. Ingress-to-Persistence Full Delivery Path
MATCH path = (r:IngressRouter)-[:DELEGATES_TO]->(h:IngressHandler)-[:INVOKES_DOMAIN]->(s:DomainService)-[:CONSUMES_PORT]->(p:RepositoryPort)-[:IMPLEMENTED_BY]->(a:RepositoryAdapter)
WHERE s.feature_name = "file_structure"
RETURN path;

// 5. Aggregated Node Counts Grouped by Architectural Role
MATCH (n)
RETURN labels(n)[0] AS architectural_label,
       count(n) AS total_count
ORDER BY total_count DESC;

// 6. Aggregated Edge Counts Grouped by Relationship Type
MATCH ()-[r]->()
RETURN type(r) AS relationship_type,
       count(r) AS total_count
ORDER BY total_count DESC;

// 7. Find Orphan Files (Nodes with 0 connections)
MATCH (n)
WHERE NOT (n)--()
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.file_path AS file_path
LIMIT 25;

// 8. Find Circular Dependencies (Cycle Detection)
MATCH (a)-[r1]->(b)-[r2]->(a)
WHERE a.id < b.id
RETURN a.id AS node_a,
       b.id AS node_b,
       type(r1) AS rel_a_to_b,
       type(r2) AS rel_b_to_a;

// --------------------------------------------------------------------------------
// PART B: PARAMETERIZED CYPHER QUERIES (FOR PYTHON ADAPTER & DRIVER)
// --------------------------------------------------------------------------------

// name: FLOW_UPSERT_NODE_CYPHER
MERGE (n:ArchitectureNode {id: $id})
SET n.label = $label,
    n.file_path = $file_path,
    n.feature_name = $feature_name,
    n += $props;

// name: FLOW_UPSERT_RELATIONSHIP_CYPHER
MATCH (a:ArchitectureNode {id: $src})
MATCH (b:ArchitectureNode {id: $tgt})
MERGE (a)-[r:DEPENDS_ON {rel_type: $rel_type}]->(b)
SET r += $props;

// name: FLOW_GET_NODE_BY_ID_CYPHER
MATCH (n {id: $id})
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.file_path AS file_path,
       n.feature_name AS feature_name,
       properties(n) AS props
LIMIT 1;

// name: FLOW_FIND_FEATURE_NODES_CYPHER
MATCH (n {feature_name: $feature_name})
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.file_path AS file_path,
       properties(n) AS props
ORDER BY n.label ASC, n.id ASC;

// name: FLOW_FIND_OUTGOING_NEIGHBORS_CYPHER
MATCH (n {id: $id})-[r]->(m)
RETURN m.id AS target_id,
       labels(m)[0] AS target_label,
       m.file_path AS target_file_path,
       type(r) AS relationship_type,
       properties(r) AS edge_props
ORDER BY target_label ASC, target_id ASC;

// name: FLOW_FIND_INCOMING_NEIGHBORS_CYPHER
MATCH (n {id: $id})<-[r]-(m)
RETURN m.id AS source_id,
       labels(m)[0] AS source_label,
       m.file_path AS source_file_path,
       type(r) AS relationship_type,
       properties(r) AS edge_props
ORDER BY source_label ASC, source_id ASC;

// name: FLOW_LIST_UPSTREAM_DEPENDENCIES_BY_TARGET_ID_CYPHER
MATCH (n)-[r]->(target)
WHERE target.id = $target_id
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.file_path AS file_path,
       type(r) AS relationship;

// name: FLOW_LIST_DOWNSTREAM_DEPENDENTS_BY_SOURCE_ID_CYPHER
MATCH (source)-[r]->(n)
WHERE source.id = $source_id
RETURN n.id AS id,
       labels(n)[0] AS label,
       n.file_path AS file_path,
       type(r) AS relationship;

// name: FLOW_RESOLVE_FEATURE_TOPOLOGY_CYPHER
MATCH (n {feature_name: $feature_name})-[r]->(m {feature_name: $feature_name})
RETURN n.id AS source_id,
       labels(n)[0] AS source_label,
       type(r) AS relationship,
       m.id AS target_id,
       labels(m)[0] AS target_label;

// name: FLOW_FIND_UPSTREAM_BLAST_RADIUS_HOP1_CYPHER
MATCH (target {id: $id})<-[r1]-(n1)
RETURN n1.id AS id,
       labels(n1)[0] AS label,
       n1.file_path AS path,
       1 AS depth,
       type(r1) AS relationship;

// name: FLOW_FIND_UPSTREAM_BLAST_RADIUS_HOP2_CYPHER
MATCH (target {id: $id})<-[r1]-(n1)<-[r2]-(n2)
RETURN n2.id AS id,
       labels(n2)[0] AS label,
       n2.file_path AS path,
       2 AS depth,
       type(r2) AS relationship;

// name: FLOW_FIND_UPSTREAM_BLAST_RADIUS_HOP3_CYPHER
MATCH (target {id: $id})<-[r1]-(n1)<-[r2]-(n2)<-[r3]-(n3)
RETURN n3.id AS id,
       labels(n3)[0] AS label,
       n3.file_path AS path,
       3 AS depth,
       type(r3) AS relationship;

// name: FLOW_FIND_DOWNSTREAM_BLAST_RADIUS_HOP1_CYPHER
MATCH (source {id: $id})-[r1]->(n1)
RETURN n1.id AS id,
       labels(n1)[0] AS label,
       n1.file_path AS path,
       1 AS depth,
       type(r1) AS relationship;

// name: FLOW_FIND_DOWNSTREAM_BLAST_RADIUS_HOP2_CYPHER
MATCH (source {id: $id})-[r1]->(n1)-[r2]->(n2)
RETURN n2.id AS id,
       labels(n2)[0] AS label,
       n2.file_path AS path,
       2 AS depth,
       type(r2) AS relationship;

// name: FLOW_DELETE_NODE_DETACH_CYPHER
MATCH (n {id: $id})
DETACH DELETE n;

// name: FLOW_CLEAR_ALL_GRAPH_CYPHER
MATCH (n)
DETACH DELETE n;

// name: FLOW_GET_PACKAGE_FEATURES_CYPHER
MATCH (p:PackageRoot {id: $id})-[:CONTAINS_FEATURE]->(s:DomainService)
RETURN s.feature_name AS name,
       s.owner AS owner,
       s.status AS status,
       s.flags AS flags,
       s.migrations AS migrations,
       s.contract_version AS contract_version;

// name: FLOW_FIND_NODE_BY_PATH_OR_ID_CYPHER
MATCH (n)
WHERE n.id = $q
   OR n.file_path = $q
   OR n.file_path ENDS WITH $q
   OR n.id ENDS WITH $q
   OR n.id CONTAINS $stem
   OR n.file_path CONTAINS $stem
RETURN n.id AS id, labels(n)[0] AS label, properties(n) AS props
LIMIT 1;

// name: FLOW_GET_NODE_PATH_CYPHER
MATCH (n)
WHERE n.id = $id OR n.id ENDS WITH $suffix
RETURN n.file_path AS path
LIMIT 1;


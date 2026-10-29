"""
Cypher queries for normalized Git Knowledge Graph and Merkle caching.
Zero string-blobs: entities and relationships are modeled as first-class graph elements.
"""

CYPHER_MERGE_IMMUTABLE_GRAPH = """
MERGE (r:GitRepository {url: $repo_url})
ON CREATE SET r.created_at = datetime()

MERGE (c:GitCommit {sha: $commit_sha, repo_url: $repo_url})
ON CREATE SET c.created_at = datetime()

MERGE (r)-[:HAS_COMMIT]->(c)

FOREACH (_ IN CASE WHEN $ref_name IS NOT NULL AND $ref_name <> '' THEN [1] ELSE [] END |
    MERGE (b:GitBranch {name: $ref_name, repo_url: $repo_url})
    ON CREATE SET b.created_at = datetime()
    MERGE (r)-[:HAS_BRANCH]->(b)
    MERGE (b)-[:POINTS_TO]->(c)
)

MERGE (e:GitEndpoint {
    target: $target,
    repo_url: $repo_url,
    http_method: $http_method,
    path: $path
})
ON CREATE SET
    e.protocol = $protocol,
    e.summary = $summary,
    e.is_deprecated = $is_deprecated,
    e.sample_curl = $sample_curl,
    e.client_code_snippet = $client_code_snippet,
    e.request_body_schema = $request_body_schema,
    e.sample_request_payload = $sample_request_payload,
    e.response_200_schema = $response_200_schema,
    e.created_at = datetime()
ON MATCH SET
    e.protocol = $protocol,
    e.summary = $summary,
    e.is_deprecated = $is_deprecated,
    e.sample_curl = $sample_curl,
    e.client_code_snippet = $client_code_snippet,
    e.request_body_schema = $request_body_schema,
    e.sample_request_payload = $sample_request_payload,
    e.response_200_schema = $response_200_schema,
    e.updated_at = datetime()

MERGE (c)-[rel:EXPOSES]->(e)
SET rel.match_type = $match_type,
    rel.contract_file = $contract_file,
    rel.updated_at = datetime()

FOREACH (_ IN CASE WHEN $auth_scheme IS NOT NULL AND $auth_scheme <> '' THEN [1] ELSE [] END |
    MERGE (a:AuthScheme {name: $auth_scheme})
    MERGE (e)-[:AUTH_BY]->(a)
)

FOREACH (dto_path IN $related_dto_files |
    MERGE (d:DataTransferObject {path: dto_path, repo_url: $repo_url})
    MERGE (e)-[:USES_DTO]->(d)
    MERGE (c)-[:CONTAINS_FILE]->(d)
)

FOREACH (_ IN CASE WHEN $contract_file IS NOT NULL AND $contract_file <> '' THEN [1] ELSE [] END |
    MERGE (k:GitContract {path: $contract_file, repo_url: $repo_url})
    MERGE (c)-[:DEFINES_CONTRACT]->(k)
    MERGE (e)-[:DEFINED_IN]->(k)
);
"""

CYPHER_RECONSTRUCT_CACHED_ENDPOINT = """
MATCH (r:GitRepository {url: $repo_url})-[:HAS_COMMIT]->(c:GitCommit {sha: $commit_sha})-[rel:EXPOSES]->(e:GitEndpoint {target: $target})
OPTIONAL MATCH (e)-[:AUTH_BY]->(a:AuthScheme)
OPTIONAL MATCH (e)-[:USES_DTO]->(d:DataTransferObject)
RETURN
    r.url AS repo_url,
    c.sha AS commit_sha,
    e.target AS target_endpoint_or_entity,
    e.http_method AS http_method,
    e.path AS path,
    e.protocol AS protocol,
    e.summary AS summary,
    e.is_deprecated AS is_deprecated,
    e.sample_curl AS sample_curl,
    e.client_code_snippet AS client_code_snippet,
    e.request_body_schema AS request_body_schema,
    e.sample_request_payload AS sample_request_payload,
    e.response_200_schema AS response_200_schema,
    a.name AS auth_scheme,
    rel.contract_file AS contract_file,
    rel.match_type AS match_type,
    collect(DISTINCT d.path) AS related_dto_files
LIMIT 1;
"""

CYPHER_INSERT_INSPECTION_RECORD = """
CREATE (l:GitInspectionLedger {
    id: $id,
    repo_url: $repo_url,
    ref_name: $ref_name,
    path_pattern: $path_pattern,
    total_found: $total_found,
    execution_ms: $execution_ms,
    created_at: datetime()
});
"""

CYPHER_GET_INSPECTION_RECORD = """
MATCH (l:GitInspectionLedger {id: $id})
RETURN
    l.id AS id,
    l.repo_url AS repo_url,
    l.ref_name AS ref_name,
    l.path_pattern AS path_pattern,
    l.total_found AS total_found,
    l.execution_ms AS execution_ms,
    toString(l.created_at) AS created_at
LIMIT 1;
"""

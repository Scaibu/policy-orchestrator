"""
Cypher queries for atomic branch pointer update and mutation ledger audit nodes in Neo4j.
"""

CYPHER_UPDATE_BRANCH_POINTER = """
MERGE (r:GitRepository {url: $repo_url})
ON CREATE SET r.created_at = datetime()

MERGE (parent:GitCommit {sha: $parent_sha, repo_url: $repo_url})
ON CREATE SET parent.created_at = datetime()
MERGE (r)-[:HAS_COMMIT]->(parent)

MERGE (c:GitCommit {sha: $new_commit_sha, repo_url: $repo_url})
ON CREATE SET c.message = $commit_message, c.created_at = datetime()
MERGE (r)-[:HAS_COMMIT]->(c)
MERGE (c)-[:PARENT]->(parent)

MERGE (b:GitBranch {name: $branch, repo_url: $repo_url})
ON CREATE SET b.created_at = datetime()
MERGE (r)-[:HAS_BRANCH]->(b)

OPTIONAL MATCH (b)-[old:POINTS_TO]->()
DELETE old
MERGE (b)-[:POINTS_TO]->(c);
"""

CYPHER_INSERT_MUTATION_LEDGER = """
CREATE (l:GitMutationLedger {
    id: $id,
    repo_url: $repo_url,
    target_branch: $target_branch,
    parent_commit_sha: $parent_commit_sha,
    new_commit_sha: $new_commit_sha,
    files_count: $files_count,
    commit_message: $commit_message,
    execution_ms: $execution_ms,
    created_at: datetime()
});
"""

CYPHER_GET_MUTATION_LEDGER = """
MATCH (l:GitMutationLedger {id: $id})
RETURN
    l.id AS id,
    l.repo_url AS repo_url,
    l.target_branch AS target_branch,
    l.parent_commit_sha AS parent_commit_sha,
    l.new_commit_sha AS new_commit_sha,
    l.files_count AS files_count,
    l.commit_message AS commit_message,
    l.execution_ms AS execution_ms,
    toString(l.created_at) AS created_at
LIMIT 1;
"""

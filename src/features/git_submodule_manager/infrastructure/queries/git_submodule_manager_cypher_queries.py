"""
Cypher queries for submodule relationship synchronization and auditing in Neo4j.
"""

CYPHER_SYNC_SUBMODULE_NODE = """
MERGE (parent:GitRepository {url: $parent_repo})
ON CREATE SET parent.created_at = datetime()

MERGE (sub:GitSubmodule {path: $submodule_path, parent_url: $parent_repo})
ON CREATE SET sub.url = $submodule_url, sub.created_at = datetime()
SET sub.gitlink_sha = $gitlink_sha, sub.updated_at = datetime()

MERGE (parent)-[:HAS_SUBMODULE]->(sub)

MERGE (child:GitRepository {url: $submodule_url})
ON CREATE SET child.created_at = datetime()

MERGE (sub)-[:POINTS_TO_REPO]->(child);
"""

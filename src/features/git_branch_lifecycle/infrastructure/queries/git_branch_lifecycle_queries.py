"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE SQL FLOW QUERIES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Named parameterized SQL queries for git_branch_lifecycle_records.
   Strictly follows naming convention: FLOW_{VERB}_{ENTITY}_{CRITERIA}.

2. INVARIANTS:
   - ZERO comments inside query constants.
================================================================================
"""

FLOW_INSERT_BRANCH_LIFECYCLE_RECORD = """
INSERT INTO git_branch_lifecycle_records (
    id, repo_url, action, source_ref, target_ref, result_sha, is_success, execution_ms, created_at
) VALUES (
    :id, :repo_url, :action, :source_ref, :target_ref, :result_sha, :is_success, :execution_ms, CURRENT_TIMESTAMP
);
"""

FLOW_SELECT_BRANCH_LIFECYCLE_RECORDS_BY_REPO = """
SELECT id, repo_url, action, source_ref, target_ref, result_sha, is_success, execution_ms, created_at
FROM git_branch_lifecycle_records
WHERE repo_url = :repo_url
ORDER BY created_at DESC
LIMIT :limit;
"""

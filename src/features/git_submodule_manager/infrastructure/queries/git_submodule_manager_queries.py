"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER PARAMETERIZED SQL FLOW QUERIES
================================================================================
1. OVERVIEW & OBJECTIVE:
   Named parameterized SQL queries for git_submodule_records.
   Adheres to naming convention: FLOW_{VERB}_{ENTITY}_{CRITERIA}.

2. INVARIANTS:
   - ZERO comments inside query constants.
================================================================================
"""

FLOW_INSERT_SUBMODULE_RECORD = """
INSERT INTO git_submodule_records (
    id, repo_url, branch, action, submodule_path, submodule_url, gitlink_sha, is_success, execution_ms, created_at
) VALUES (
    :id, :repo_url, :branch, :action, :submodule_path, :submodule_url, :gitlink_sha, :is_success, :execution_ms, CURRENT_TIMESTAMP
);
"""

FLOW_SELECT_SUBMODULE_RECORD_BY_ID = """
SELECT id, repo_url, branch, action, submodule_path, submodule_url, gitlink_sha, is_success, execution_ms, created_at
FROM git_submodule_records
WHERE id = :id;
"""

FLOW_SELECT_SUBMODULE_RECORDS_BY_REPO = """
SELECT id, repo_url, branch, action, submodule_path, submodule_url, gitlink_sha, is_success, execution_ms, created_at
FROM git_submodule_records
WHERE repo_url = :repo_url
ORDER BY created_at DESC
LIMIT :limit;
"""

"""
================================================================================
BLUEPRINT: GIT VERIFIER PARAMETERIZED SQL QUERIES
================================================================================
1. OVERVIEW:
   Named parameterized SQL queries for recording and auditing Git verification.
================================================================================
"""

FLOW_INSERT_GIT_VERIFICATION_RECORD = """
INSERT INTO git_verification_records (id, repo_path, is_clean, submodules_count, execution_ms, created_at)
VALUES (%(id)s, %(repo_path)s, %(is_clean)s, %(submodules_count)s, %(execution_ms)s, NOW())
ON CONFLICT (id) DO UPDATE SET
    is_clean = EXCLUDED.is_clean,
    submodules_count = EXCLUDED.submodules_count,
    execution_ms = EXCLUDED.execution_ms;
"""

FLOW_GET_GIT_VERIFICATION_RECORD_BY_ID = """
SELECT id, repo_path, is_clean, submodules_count, execution_ms, created_at
FROM git_verification_records
WHERE id = %(id)s;
"""

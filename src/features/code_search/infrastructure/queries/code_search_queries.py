"""
================================================================================
BLUEPRINT: CODE SEARCH PARAMETERIZED SQL QUERIES
================================================================================
1. OVERVIEW:
   Named parameterized SQL queries for recording and auditing code search operations.
================================================================================
"""

FLOW_INSERT_CODE_SEARCH_RECORD = """
INSERT INTO code_search_records (id, pattern_count, matches_found, execution_ms, created_at)
VALUES (%(id)s, %(pattern_count)s, %(matches_found)s, %(execution_ms)s, NOW())
ON CONFLICT (id) DO UPDATE SET
    matches_found = EXCLUDED.matches_found,
    execution_ms = EXCLUDED.execution_ms;
"""

FLOW_GET_CODE_SEARCH_RECORD_BY_ID = """
SELECT id, pattern_count, matches_found, execution_ms, created_at
FROM code_search_records
WHERE id = %(id)s;
"""

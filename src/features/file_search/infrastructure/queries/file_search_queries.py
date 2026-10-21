"""
================================================================================
BLUEPRINT: FILE SEARCH PARAMETERIZED SQL QUERIES
================================================================================
1. OVERVIEW:
   Named parameterized SQL queries for recording and auditing file search metrics.
2. ZERO-INLINE-COMMENT DOCTRINE:
   All queries are constants documented in this top header block.
================================================================================
"""

FLOW_INSERT_FILE_SEARCH_RECORD = """
INSERT INTO file_search_records (id, root_path, total_found, execution_ms, created_at)
VALUES (%(id)s, %(root_path)s, %(total_found)s, %(execution_ms)s, NOW())
ON CONFLICT (id) DO UPDATE SET
    total_found = EXCLUDED.total_found,
    execution_ms = EXCLUDED.execution_ms;
"""

FLOW_GET_FILE_SEARCH_RECORD_BY_ID = """
SELECT id, root_path, total_found, execution_ms, created_at
FROM file_search_records
WHERE id = %(id)s;
"""

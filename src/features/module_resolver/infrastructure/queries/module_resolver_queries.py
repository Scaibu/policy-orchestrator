"""
================================================================================
BLUEPRINT: MODULE RESOLVER PARAMETERIZED SQL QUERIES
================================================================================
1. OVERVIEW:
   Named parameterized SQL queries for recording and auditing module resolution.
================================================================================
"""

FLOW_INSERT_MODULE_RESOLUTION_RECORD = """
INSERT INTO module_resolution_records (id, module_name, is_valid, collision_detected, execution_ms, created_at)
VALUES (%(id)s, %(module_name)s, %(is_valid)s, %(collision_detected)s, %(execution_ms)s, NOW())
ON CONFLICT (id) DO UPDATE SET
    is_valid = EXCLUDED.is_valid,
    collision_detected = EXCLUDED.collision_detected,
    execution_ms = EXCLUDED.execution_ms;
"""

FLOW_GET_MODULE_RESOLUTION_RECORD_BY_ID = """
SELECT id, module_name, is_valid, collision_detected, execution_ms, created_at
FROM module_resolution_records
WHERE id = %(id)s;
"""

FLOW_INSERT_REMOTE_INSPECTION_RECORD = """
INSERT INTO git_remote_inspection_records (
    id,
    repo_url,
    ref_name,
    path_pattern,
    total_found,
    execution_ms,
    created_at
) VALUES (
    :id,
    :repo_url,
    :ref_name,
    :path_pattern,
    :total_found,
    :execution_ms,
    NOW()
);
"""

FLOW_SELECT_REMOTE_INSPECTION_RECORD_BY_ID = """
SELECT
    id,
    repo_url,
    ref_name,
    path_pattern,
    total_found,
    execution_ms,
    created_at
FROM git_remote_inspection_records
WHERE id = :id;
"""

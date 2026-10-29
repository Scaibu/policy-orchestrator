FLOW_INSERT_REMOTE_MUTATION_RECORD = """
INSERT INTO git_remote_mutation_records (
    id,
    repo_url,
    target_branch,
    parent_commit_sha,
    new_commit_sha,
    files_count,
    commit_message,
    execution_ms,
    created_at
) VALUES (
    :id,
    :repo_url,
    :target_branch,
    :parent_commit_sha,
    :new_commit_sha,
    :files_count,
    :commit_message,
    :execution_ms,
    NOW()
);
"""

FLOW_SELECT_REMOTE_MUTATION_RECORD_BY_ID = """
SELECT
    id,
    repo_url,
    target_branch,
    parent_commit_sha,
    new_commit_sha,
    files_count,
    commit_message,
    execution_ms,
    created_at
FROM git_remote_mutation_records
WHERE id = :id;
"""

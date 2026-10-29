-- Migration 0006: Create git_branch_lifecycle_records table (PostgreSQL / AlloyDB)
CREATE TABLE IF NOT EXISTS git_branch_lifecycle_records (
    id TEXT PRIMARY KEY CONSTRAINT git_branch_lifecycle_records_id_pk PRIMARY KEY,
    repo_url TEXT NOT NULL CONSTRAINT git_branch_lifecycle_records_repo_url_not_empty_ck CHECK (repo_url = btrim(repo_url) AND repo_url <> ''),
    action TEXT NOT NULL CONSTRAINT git_branch_lifecycle_records_action_ck CHECK (action IN ('CREATE_BRANCH', 'MERGE_BRANCH', 'DELETE_BRANCH', 'CREATE_REPO')),
    source_ref TEXT,
    target_ref TEXT,
    result_sha TEXT,
    is_success BOOLEAN NOT NULL DEFAULT TRUE,
    execution_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0 CONSTRAINT git_branch_lifecycle_records_execution_ms_nonneg_ck CHECK (execution_ms >= 0.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS git_branch_lifecycle_records_repo_url_idx ON git_branch_lifecycle_records (repo_url);
CREATE INDEX IF NOT EXISTS git_branch_lifecycle_records_action_idx ON git_branch_lifecycle_records (action);
CREATE INDEX IF NOT EXISTS git_branch_lifecycle_records_created_at_idx ON git_branch_lifecycle_records (created_at DESC);

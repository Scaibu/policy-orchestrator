-- Migration 0006: Create git_branch_lifecycle_records table (SQLite)
CREATE TABLE IF NOT EXISTS git_branch_lifecycle_records (
    id TEXT PRIMARY KEY,
    repo_url TEXT NOT NULL CHECK (length(trim(repo_url)) > 0),
    action TEXT NOT NULL CHECK (action IN ('CREATE_BRANCH', 'MERGE_BRANCH', 'DELETE_BRANCH', 'CREATE_REPO')),
    source_ref TEXT,
    target_ref TEXT,
    result_sha TEXT,
    is_success INTEGER NOT NULL DEFAULT 1 CHECK (is_success IN (0, 1)),
    execution_ms REAL NOT NULL DEFAULT 0.0 CHECK (execution_ms >= 0.0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_git_branch_lifecycle_repo_url ON git_branch_lifecycle_records (repo_url);
CREATE INDEX IF NOT EXISTS idx_git_branch_lifecycle_action ON git_branch_lifecycle_records (action);
CREATE INDEX IF NOT EXISTS idx_git_branch_lifecycle_created_at ON git_branch_lifecycle_records (created_at DESC);

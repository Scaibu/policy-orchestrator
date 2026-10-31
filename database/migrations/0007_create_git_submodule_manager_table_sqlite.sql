-- Migration 0007: Create git_submodule_records table (SQLite)
CREATE TABLE IF NOT EXISTS git_submodule_records (
    id TEXT PRIMARY KEY,
    repo_url TEXT NOT NULL CHECK (length(trim(repo_url)) > 0),
    branch TEXT NOT NULL CHECK (length(trim(branch)) > 0),
    action TEXT NOT NULL CHECK (action IN ('ADD_SUBMODULE', 'UPDATE_SUBMODULE', 'PULL_SUBMODULES', 'PUSH_MULTI_REPO', 'SYNC_STATUS')),
    submodule_path TEXT NOT NULL CHECK (length(trim(submodule_path)) > 0),
    submodule_url TEXT,
    gitlink_sha TEXT,
    is_success INTEGER NOT NULL DEFAULT 1 CHECK (is_success IN (0, 1)),
    execution_ms REAL NOT NULL DEFAULT 0.0 CHECK (execution_ms >= 0.0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_git_submodule_repo_url ON git_submodule_records (repo_url);
CREATE INDEX IF NOT EXISTS idx_git_submodule_action ON git_submodule_records (action);
CREATE INDEX IF NOT EXISTS idx_git_submodule_created_at ON git_submodule_records (created_at DESC);

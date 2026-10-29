-- migration:      0005
-- description:    create git remote mutation records table for SQLite with strict explicit constraints
-- author:         Antigravity / Lead Platform Architect
-- date:           2026-10-29
-- reversible:     YES

CREATE TABLE IF NOT EXISTS git_remote_mutation_records (
    id TEXT PRIMARY KEY,
    repo_url TEXT NOT NULL CHECK (length(trim(repo_url)) > 0),
    target_branch TEXT NOT NULL CHECK (length(trim(target_branch)) > 0),
    parent_commit_sha TEXT NOT NULL CHECK (length(parent_commit_sha) = 40),
    new_commit_sha TEXT NOT NULL CHECK (length(new_commit_sha) = 40),
    files_count INTEGER NOT NULL DEFAULT 1 CHECK (files_count >= 1),
    commit_message TEXT NOT NULL CHECK (length(trim(commit_message)) > 0),
    execution_ms REAL NOT NULL DEFAULT 0.0 CHECK (execution_ms >= 0.0),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS git_remote_mutation_records_repo_idx ON git_remote_mutation_records(repo_url);
CREATE INDEX IF NOT EXISTS git_remote_mutation_records_branch_idx ON git_remote_mutation_records(target_branch);

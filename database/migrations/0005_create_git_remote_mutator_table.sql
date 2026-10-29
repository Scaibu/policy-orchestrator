-- migration:      0005
-- description:    create git remote mutation records table for audit ledger with strict explicit constraints
-- author:         Antigravity / Lead Platform Architect
-- date:           2026-10-29
-- reversible:     YES

CREATE TABLE IF NOT EXISTS git_remote_mutation_records (
    id TEXT CONSTRAINT git_remote_mutation_records_id_pk PRIMARY KEY,
    repo_url TEXT NOT NULL CONSTRAINT git_remote_mutation_records_repo_url_not_empty_ck CHECK (repo_url = btrim(repo_url) AND repo_url <> ''),
    target_branch TEXT NOT NULL CONSTRAINT git_remote_mutation_records_target_branch_not_empty_ck CHECK (target_branch = btrim(target_branch) AND target_branch <> ''),
    parent_commit_sha TEXT NOT NULL CONSTRAINT git_remote_mutation_records_parent_sha_ck CHECK (length(parent_commit_sha) = 40 AND parent_commit_sha ~ '^[0-9a-fA-F]{40}$'),
    new_commit_sha TEXT NOT NULL CONSTRAINT git_remote_mutation_records_new_sha_ck CHECK (length(new_commit_sha) = 40 AND new_commit_sha ~ '^[0-9a-fA-F]{40}$'),
    files_count INTEGER NOT NULL DEFAULT 1 CONSTRAINT git_remote_mutation_records_files_count_ck CHECK (files_count >= 1),
    commit_message TEXT NOT NULL CONSTRAINT git_remote_mutation_records_commit_msg_not_empty_ck CHECK (commit_message = btrim(commit_message) AND commit_message <> ''),
    execution_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0 CONSTRAINT git_remote_mutation_records_exec_ms_ck CHECK (execution_ms >= 0.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS git_remote_mutation_records_repo_idx ON git_remote_mutation_records(repo_url);
CREATE INDEX IF NOT EXISTS git_remote_mutation_records_branch_idx ON git_remote_mutation_records(target_branch);

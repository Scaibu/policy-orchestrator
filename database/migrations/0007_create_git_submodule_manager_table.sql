-- Migration 0007: Create git_submodule_records table (PostgreSQL / AlloyDB)
CREATE TABLE IF NOT EXISTS git_submodule_records (
    id TEXT PRIMARY KEY CONSTRAINT git_submodule_records_id_pk PRIMARY KEY,
    repo_url TEXT NOT NULL CONSTRAINT git_submodule_records_repo_url_not_empty_ck CHECK (repo_url = btrim(repo_url) AND repo_url <> ''),
    branch TEXT NOT NULL CONSTRAINT git_submodule_records_branch_not_empty_ck CHECK (branch = btrim(branch) AND branch <> ''),
    action TEXT NOT NULL CONSTRAINT git_submodule_records_action_ck CHECK (action IN ('ADD_SUBMODULE', 'UPDATE_SUBMODULE', 'PULL_SUBMODULES', 'PUSH_MULTI_REPO', 'SYNC_STATUS')),
    submodule_path TEXT NOT NULL CONSTRAINT git_submodule_records_sub_path_not_empty_ck CHECK (submodule_path = btrim(submodule_path) AND submodule_path <> ''),
    submodule_url TEXT,
    gitlink_sha TEXT,
    is_success BOOLEAN NOT NULL DEFAULT TRUE,
    execution_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0 CONSTRAINT git_submodule_records_execution_ms_nonneg_ck CHECK (execution_ms >= 0.0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS git_submodule_records_repo_url_idx ON git_submodule_records (repo_url);
CREATE INDEX IF NOT EXISTS git_submodule_records_action_idx ON git_submodule_records (action);
CREATE INDEX IF NOT EXISTS git_submodule_records_created_at_idx ON git_submodule_records (created_at DESC);

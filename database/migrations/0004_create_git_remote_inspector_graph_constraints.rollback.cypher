// migration:      0004_rollback
// description:    drop git remote inspector knowledge graph unique constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-29
// reversible:     YES

DROP CONSTRAINT git_repository_url_unique IF EXISTS;
DROP CONSTRAINT git_commit_sha_repo_unique IF EXISTS;
DROP CONSTRAINT git_branch_name_repo_unique IF EXISTS;
DROP CONSTRAINT git_endpoint_target_repo_unique IF EXISTS;
DROP CONSTRAINT auth_scheme_name_unique IF EXISTS;
DROP CONSTRAINT git_inspection_ledger_id_unique IF EXISTS;

DROP INDEX git_contract_path_idx IF EXISTS;
DROP INDEX data_transfer_object_path_idx IF EXISTS;

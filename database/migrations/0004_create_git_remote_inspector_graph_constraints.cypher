// migration:      0004
// description:    create git remote inspector knowledge graph unique constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-29
// reversible:     YES

CREATE CONSTRAINT git_repository_url_unique IF NOT EXISTS FOR (r:GitRepository) REQUIRE r.url IS UNIQUE;
CREATE CONSTRAINT git_commit_sha_repo_unique IF NOT EXISTS FOR (c:GitCommit) REQUIRE (c.sha, c.repo_url) IS UNIQUE;
CREATE CONSTRAINT git_branch_name_repo_unique IF NOT EXISTS FOR (b:GitBranch) REQUIRE (b.name, b.repo_url) IS UNIQUE;
CREATE CONSTRAINT git_endpoint_target_repo_unique IF NOT EXISTS FOR (e:GitEndpoint) REQUIRE (e.target, e.repo_url) IS UNIQUE;
CREATE CONSTRAINT auth_scheme_name_unique IF NOT EXISTS FOR (a:AuthScheme) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT git_inspection_ledger_id_unique IF NOT EXISTS FOR (l:GitInspectionLedger) REQUIRE l.id IS UNIQUE;

CREATE INDEX git_contract_path_idx IF NOT EXISTS FOR (k:GitContract) ON (k.path);
CREATE INDEX data_transfer_object_path_idx IF NOT EXISTS FOR (d:DataTransferObject) ON (d.path);

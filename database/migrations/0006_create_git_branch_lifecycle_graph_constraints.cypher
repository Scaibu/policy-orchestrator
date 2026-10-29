// migration:      0006
// description:    create git branch lifecycle graph constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-29
// reversible:     YES

CREATE CONSTRAINT git_branch_repo_name_unique IF NOT EXISTS FOR (b:GitBranch) REQUIRE (b.repo, b.name) IS UNIQUE;
CREATE INDEX git_branch_name_idx IF NOT EXISTS FOR (b:GitBranch) ON (b.name);

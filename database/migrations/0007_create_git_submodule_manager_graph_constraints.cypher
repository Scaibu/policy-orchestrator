// migration:      0007
// description:    create git submodule manager graph constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-31
// reversible:     YES

CREATE CONSTRAINT git_submodule_parent_path_unique IF NOT EXISTS FOR (s:GitSubmodule) REQUIRE (s.parent_url, s.path) IS UNIQUE;
CREATE INDEX git_submodule_path_idx IF NOT EXISTS FOR (s:GitSubmodule) ON (s.path);

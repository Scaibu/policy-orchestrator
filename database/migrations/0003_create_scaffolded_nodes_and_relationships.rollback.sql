-- rollback:       0003
-- description:    drop scaffolded files, folders, and relationships ledger
-- author:         Antigravity / Lead Software Engineer
-- date:           2026-10-12
-- reversible:     YES

DROP TABLE IF EXISTS scaffolded_relationships;
DROP TABLE IF EXISTS scaffolded_nodes;

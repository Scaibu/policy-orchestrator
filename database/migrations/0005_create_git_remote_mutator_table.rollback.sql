-- migration:      0005
-- description:    rollback git remote mutation records table
-- author:         Antigravity / Lead Platform Architect
-- date:           2026-10-29
-- reversible:     YES

DROP TABLE IF EXISTS git_remote_mutation_records CASCADE;

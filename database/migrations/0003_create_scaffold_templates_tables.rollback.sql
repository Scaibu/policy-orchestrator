-- migration:      0003.rollback
-- description:    rollback scaffold templates, features, and packages tables
-- author:         Antigravity / Lead Software Engineer
-- date:           2026-10-12
-- depends_on:     NONE
-- reversible:     YES
-- lock_risk:      LOW

DROP TABLE IF EXISTS scaffolded_packages CASCADE;
DROP TABLE IF EXISTS scaffolded_features CASCADE;
DROP TABLE IF EXISTS scaffold_templates CASCADE;

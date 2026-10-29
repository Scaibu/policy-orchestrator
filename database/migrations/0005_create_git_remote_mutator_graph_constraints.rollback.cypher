// migration:      0005
// description:    rollback git remote mutator knowledge graph unique constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-29
// reversible:     YES

DROP CONSTRAINT git_mutation_ledger_id_unique IF EXISTS;

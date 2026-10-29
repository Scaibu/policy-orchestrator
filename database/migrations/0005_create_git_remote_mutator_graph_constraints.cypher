// migration:      0005
// description:    create git remote mutator knowledge graph unique constraints and indexes
// author:         Antigravity / Lead Platform Architect
// date:           2026-10-29
// reversible:     YES

CREATE CONSTRAINT git_mutation_ledger_id_unique IF NOT EXISTS FOR (l:GitMutationLedger) REQUIRE l.id IS UNIQUE;

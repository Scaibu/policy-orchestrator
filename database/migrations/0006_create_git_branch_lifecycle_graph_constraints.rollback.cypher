-- Rollback Migration 0006: Neo4j Constraints
DROP CONSTRAINT git_branch_repo_name_unique IF EXISTS;
DROP INDEX git_branch_name_idx IF EXISTS;

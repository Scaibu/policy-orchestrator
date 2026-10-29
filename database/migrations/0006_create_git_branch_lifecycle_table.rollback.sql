-- Rollback Migration 0006: Drop git_branch_lifecycle_records table (PostgreSQL / AlloyDB)
DROP TABLE IF EXISTS git_branch_lifecycle_records CASCADE;

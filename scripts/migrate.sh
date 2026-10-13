#!/usr/bin/env bash
# ==============================================================================
# UNIFIED DATABASE & GRAPH MIGRATION RUNNER
# Applies AlloyDB (PostgreSQL) DDL and Neo4j Graph DB constraints & indexes.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTION="${1:-run}"
DB_TARGET="${2:-all}"

echo "================================================================================"
echo "RUNNING MIGRATIONS: Action=${ACTION}, Target=${DB_TARGET}"
echo "================================================================================"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python3 -c "
import sys
from src.infra.adapters.database.migration_runner import DatabaseMigrationRunner
from src.infra.config.config_loader import AppConfigLoader

action = '${ACTION}'
db_target = '${DB_TARGET}'

db_url = AppConfigLoader.get_database_url()
runner = DatabaseMigrationRunner(db_url)

if action == 'run':
    if db_target in ('all', 'alloydb', 'sql', 'postgres'):
        print('--> Applying AlloyDB (PostgreSQL) Migrations...')
        sql_res = runner.run_migrations()
        print('    AlloyDB Status:', sql_res)
    if db_target in ('all', 'graph', 'neo4j'):
        print('--> Applying Neo4j Knowledge Graph Constraints & Indexes...')
        graph_res = runner.run_graph_migrations()
        print('    Neo4j Status:', graph_res)
    print('✅ Migrations completed successfully.')

elif action == 'rollback':
    if db_target in ('all', 'graph', 'neo4j'):
        print('--> Rolling back Neo4j Graph Constraints...')
        graph_res = runner.rollback_graph_migrations()
        print('    Neo4j Status:', graph_res)
    if db_target in ('all', 'alloydb', 'sql', 'postgres'):
        print('--> Rolling back AlloyDB Migrations...')
        sql_res = runner.rollback_migrations()
        print('    AlloyDB Status:', sql_res)
    print('🔄 Rollback completed.')

else:
    print(f'Unknown action: {action}')
    sys.exit(1)
"

echo "================================================================================"
echo "MIGRATION OPERATION COMPLETED"
echo "================================================================================"

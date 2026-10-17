# Policy Orchestrator (`policy-orchestrator`)

> **Declarative Microservice Scaffolding & Architectural Dependency Engine**  
> Enforces Rule 0 (Sub-Package Isolation) and Rule 1 (Hexagonal & Triad Features) across polyglot repositories with dual AlloyDB and Neo4j graph lineage tracking.

---

## How to Use

### 1. Scaffold a Complete Service Package (Rule 0)
Creates the standard production package directory tree (`contracts/`, `config/`, `database/`, `src/shared/`, `src/infra/`, `src/api/`, `src/features/`):
```bash
python3 src/api/cli/folder_scaffold_cli.py scaffold-package \
  --name <package-name> \
  --base-dir <base-dir>
```

### 2. Scaffold a Feature Inside a Package (Rule 1)
- **Backend Hexagonal Feature** (`service/`, `repository/`, `queries/`, `rules/`, `machines/`, `types/`, `tests/`):
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py scaffold-feature \
    --name <feature-name> \
    --package <package-name> \
    --type backend \
    --base-dir <base-dir>
  ```
- **Frontend Triad Feature** (State, Controller, Model):
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py scaffold-feature \
    --name <feature-name> \
    --package <package-name> \
    --type frontend \
    --base-dir <base-dir>
  ```

### 3. Scaffold Consolidated Database Hub
Creates the unified `database/` structure (`migrations/sql/`, `migrations/cypher/`, `seeds/`, `engine/`, `config/`, and `schema.lock`):
```bash
python3 src/api/cli/folder_scaffold_cli.py scaffold-database \
  --package <package-name> \
  --base-dir <base-dir>
```

### 4. Scaffold Consolidated Messaging Hub
Creates the unified `messaging/` structure (`topics/`, `schemas/`, `policies/`, `engine/client/`, `engine/producers/`, `engine/consumers/`, `engine/pipeline/`, and `topics.lock`):
```bash
python3 src/api/cli/folder_scaffold_cli.py scaffold-messaging \
  --package <package-name> \
  --base-dir <base-dir>
```

### 5. Analyze Architectural Knowledge Graph
- **Blast Radius / Impact Analysis** (find callers and dependencies affected by a change):
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py analyze-impact \
    --target <target-file-or-node-id>
  ```
- **Lineage Tracing** (trace dependency chain from API handlers to database adapters):
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py analyze-lineage \
    --file <file-path>
  ```

### 4. Run Database & Graph Migrations
```bash
# Run all SQL and Neo4j graph migrations
python3 src/api/cli/main.py migrate --db all

# Rollback migrations
python3 src/api/cli/main.py migrate --action rollback --db all
```

### 5. Start the REST API Server
```bash
# Start local server via CLI
python3 src/api/cli/main.py serve --host 0.0.0.0 --port 8000
```

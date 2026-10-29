# Policy Orchestrator (`policy-orchestrator`)

> **Declarative Microservice Scaffolding, Zero-Clone Git Intelligence & Architectural Dependency Engine**  
> Enforces Rule 0 (Sub-Package Isolation) and Rule 1 (Hexagonal & Triad Features) across polyglot repositories with dual AlloyDB and Neo4j graph lineage tracking.

---

## Running the Server

### 1. Launch FastAPI Server
Run the production server via Uvicorn factory:
```bash
# Factory creation with custom host and port
python3 -m uvicorn src.api.rest.app:create_app --factory --host 127.0.0.1 --port 8089

# Or via CLI entrypoint
python3 src/api/cli/main.py serve --host 127.0.0.1 --port 8089
```

Health check verification:
```bash
curl -s http://127.0.0.1:8089/api/v1/health | jq .
```

---

## Zero-Clone Remote Git Features

### Feature 1: Git Remote Inspector (`git_remote_inspector`)
Inspect remote Git repositories without local disk cloning. Supports file search, recursive tree retrieval, branch listing, contract discovery, static call chain tracing, and one-shot microservice integration context with Neo4j Merkle caching.

#### A. One-Shot Microservice Integration (`/integrate-context`)
Resolves OpenAPI contracts, dereferences multi-hop `` schemas, generates runnable client snippets, extracts auth schemes, or falls back to polyglot AST call chains:
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-inspector/integrate-context   -H "Content-Type: application/json"   -H "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"   -H "x-request-id: req-integ-1"   -H "x-correlation-id: corr-integ-1"   -d '{
    "repo_url": "https://github.com/Scaibu/policy-orchestrator",
    "endpoint_or_entity": "/api/v1/file-search/scan",
    "ref_name": "main"
  }' | jq .
```

#### B. Search Remote Files by Pattern (`/search`)
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-inspector/search   -H "Content-Type: application/json"   -d '{
    "repo_url": "https://github.com/Scaibu/policy-orchestrator",
    "ref_name": "main",
    "path_pattern": "*.py"
  }' | jq .
```

#### C. Discover API & Data Contracts (`/discover-contracts`)
Identifies OpenAPI specs, Protobuf schemas, routing files, and DTOs:
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-inspector/discover-contracts   -H "Content-Type: application/json"   -d '{
    "repo_url": "https://github.com/Scaibu/policy-orchestrator",
    "ref_name": "main"
  }' | jq .
```

#### D. CLI Command for Remote Inspection
```bash
python3 src/api/cli/main.py remote-inspect search   --repo Scaibu/policy-orchestrator   --branch main   --pattern "*.py"
```

---

### Feature 2: Git Remote Mutator (`git_remote_mutator`)
Enables atomic remote Git mutations (single file edits, batch multi-file modifications, and sparse file deletions) via low-level Git Merkle plumbing (`blobs` → `trees` → `commits` → `refs CAS`) with Optimistic Concurrency Control (OCC) and live Neo4j branch pointer synchronization (`(:GitBranch)-[:POINTS_TO]->(:GitCommit)`).

#### A. Mutate a Single Remote File (`/mutate-file`)
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-mutator/mutate-file   -H "Content-Type: application/json"   -H "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"   -H "x-request-id: req-mut-1"   -H "x-correlation-id: corr-mut-1"   -d '{
    "repo_url": "Scaibu/policy-orchestrator",
    "branch": "main",
    "file_path": "src/config.json",
    "content": "{"active": true, "version": "2.0"}",
    "commit_message": "feat: update config via zero-clone mutator",
    "expected_parent_sha": "<optional-current-head-sha-for-occ>"
  }' | jq .
```

#### B. Atomic Multi-File Batch Mutation (`/mutate-batch`)
Applies modifications and deletions in a single atomic Git tree without intermediate commits:
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-mutator/mutate-batch   -H "Content-Type: application/json"   -H "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"   -H "x-request-id: req-mut-batch"   -H "x-correlation-id: corr-mut-batch"   -d '{
    "repo_url": "Scaibu/policy-orchestrator",
    "branch": "main",
    "mutations": [
      {
        "path": "README.md",
        "content": "# Updated README
New content here",
        "is_deleted": false
      },
      {
        "path": "deprecated_module.py",
        "is_deleted": true
      }
    ],
    "commit_message": "chore: atomic multi-file update and prune"
  }' | jq .
```

#### C. Delete a Remote File (`/delete-file`)
```bash
curl -X POST http://127.0.0.1:8089/api/v1/git-remote-mutator/delete-file   -H "Content-Type: application/json"   -H "traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"   -d '{
    "repo_url": "Scaibu/policy-orchestrator",
    "branch": "main",
    "file_path": "temp_file.txt",
    "commit_message": "chore: remove temporary file"
  }' | jq .
```

#### D. CLI Command for Remote Mutation
```bash
# Mutate a single file via CLI
python3 src/api/cli/main.py remote-mutate file   --repo Scaibu/policy-orchestrator   --branch main   --path src/config.json   --content '{"key": "value"}'   --message "feat: cli mutation"

# Delete a file via CLI
python3 src/api/cli/main.py remote-mutate delete   --repo Scaibu/policy-orchestrator   --branch main   --path temp_file.txt   --message "chore: delete temp file"
```

---

## Core Scaffolding & Migration Commands

### 1. Scaffold a Complete Service Package (Rule 0)
```bash
python3 src/api/cli/folder_scaffold_cli.py scaffold-package   --name <package-name>   --base-dir <base-dir>
```

### 2. Scaffold a Feature Inside a Package (Rule 1)
- **Backend Hexagonal Feature**:
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py scaffold-feature     --name <feature-name>     --package <package-name>     --type backend     --base-dir <base-dir>
  ```
- **Frontend Triad Feature**:
  ```bash
  python3 src/api/cli/folder_scaffold_cli.py scaffold-feature     --name <feature-name>     --package <package-name>     --type frontend     --base-dir <base-dir>
  ```

### 3. Run Database & Graph Migrations
```bash
# Run all SQL and Neo4j graph migrations
python3 src/api/cli/main.py migrate --db all

# Run Neo4j graph migrations only
python3 src/api/cli/main.py migrate --db neo4j
```

---

## Architectural Decision Records (ADR)
- [ADR-0004: Zero-Clone Remote Microservice Integration Intelligence Engine](docs/adr/ADR-0004-zero-clone-remote-microservice-integration-engine.md)
- [ADR-0005: Zero-Clone Remote Git Mutation Engine](docs/adr/ADR-0005-zero-clone-remote-git-mutation-engine.md)

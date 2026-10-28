# ADR-0003: Zero-Clone Remote Git Operations via Git Database Provider API

| Field | Value |
| :--- | :--- |
| **Status** | Accepted |
| **Date** | 2026-10-28 |
| **Deciders** | Core Architecture Team, Policy Orchestrator Leads |
| **Related Rules** | [1-feature-folder-writing-rule.md](../../rules/folder-structure/1-feature-folder-writing-rule.md), [2-apis.writing.rule.md](../../rules/folder-structure/2-apis.writing.rule.md), [4-functions-and-classes-writing-rule.md](../../rules/folder-structure/4-functions-and-classes-writing-rule.md) |
| **Implementation** | `src/features/git_remote_inspector`, `git_remote_mutator`, `git_branch_lifecycle` |

---

## 1. Context

Operating agents and policy orchestrators across large enterprise repositories often requires inspecting, editing, or committing to specific files or directories without the overhead of full repository cloning. 

Standard git operations (`git clone`, `git checkout`, `git push`) present severe architectural bottlenecks:
1. **Disk & Inode Exhaustion:** Enterprise monorepos often exceed hundreds of megabytes or gigabytes. Cloning exhausts container disk quotas and inode tables.
2. **Cold-Start Latency:** A standard clone across multi-GB packfiles can take 30–120 seconds, violating our sub-second API execution SLA.
3. **Dirty Workspace & Concurrency Hazards:** Ephemeral agent runtimes running concurrent tasks risk race conditions, dirty working trees, and orphaned index locks.
4. **Token Isolation Hazards:** Sharing local git credentials or writing tokens into local `.git/config` creates privilege escalation vulnerabilities.

We require a stateless, hexagonal architecture that operates directly against remote Git objects without creating a local working tree.

---

## 2. Decision Drivers

| Driver | Why It Matters |
| :--- | :--- |
| **Zero Disk Footprint** | Must perform read, search, diff, and mutation without writing repository trees or packfiles to local disk. |
| **Sub-Second Execution (<500ms)** | Fetching trees and reading blobs via structured JSON APIs executes in ~200ms vs >30s for clone. |
| **Atomic Concurrency & ETag Verification** | Must support optimistic concurrency control against parent commit SHAs to prevent blind overwrites. |
| **Hexagonal Inversion of Control** | Domain services must never import HTTP clients or git drivers directly; must rely strictly on `GitRemoteClientPort`. |
| **W3C Trace Context Propagation** | Every outbound call must inject `traceparent` headers to preserve end-to-end distributed observability. |

---

## 3. Considered Options

| Option | Pros | Cons | Decision |
| :--- | :--- | :--- | :--- |
| **Option A: Full Git Clone (`git clone --depth 1`)** | Native Git CLI commands. | High latency (>30s), disk quota exhaustion, requires disk cleanup scripts, non-atomic. | Rejected |
| **Option B: Sparse-Checkout Partial Clone (`--filter=blob:none`)** | Only downloads referenced blobs. | Still requires `.git` object directory initialization, local process execution, disk I/O. | Retained as secondary fallback adapter |
| **Option C: Remote Git Data API (Blobs, Trees, Commits, Refs)** | Zero disk footprint, memory-only execution, atomic Merkle tree updates, sub-250ms latency. | Requires Git hosting provider API (GitHub / GitLab REST/GraphQL). | **Accepted (Primary)** |

---

## 4. Decision Outcome

**Chosen option: Option C (Remote Git Data API via Hexagonal Driven Port)**

### Execution Workflow

```
[Client Request]
       │
       ▼
(01) Ingress Router -> (02) Handler (W3C traceparent extraction, root span)
       │
       ▼
(03) ACL Schema -> (04) Domain DTO (Validation with extra='forbid')
       │
       ▼
(06) Domain Service (Child span, deterministic state machine, try/catch/finally)
       │
       ▼
(10) GitRemoteClientPort (Universal driven port)
       │
       ▼
(11) GitHubRemoteClientAdapter (Decorated outbound HTTP client)
       │
       ├─► GET /repos/{owner}/{repo}/git/trees/{sha}?recursive=1  (Zero-clone tree indexing)
       ├─► GET /repos/{owner}/{repo}/git/blobs/{sha}             (Raw blob extraction)
       ├─► POST /repos/{owner}/{repo}/git/blobs                  (Remote blob creation)
       ├─► POST /repos/{owner}/{repo}/git/trees                  (Sparse tree assembly)
       ├─► POST /repos/{owner}/{repo}/git/commits                (Signed commit creation)
       └─► PATCH /repos/{owner}/{repo}/git/refs/heads/{branch}   (Atomic ref update)
```

---

## 5. Consequences

| Positive | Negative / Risk Accepted |
| :--- | :--- |
| Zero disk storage required on host containers. | Depends on Git provider REST API availability and rate limits (5,000 req/hr per token). |
| Latency reduced from 45s clone down to ~200ms per tree traversal. | High-frequency polling requires token caching or GitHub App installations. |
| Elimination of local file cleanup and orphaned `.git` lock files. | Non-GitHub/GitLab bare git servers require Git CLI sparse-checkout adapter. |
| Atomic, race-free branch updates via optimistic parent commit SHA validation. | |

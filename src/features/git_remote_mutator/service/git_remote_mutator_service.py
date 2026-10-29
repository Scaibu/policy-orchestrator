"""
================================================================================
BLUEPRINT: GIT REMOTE MUTATOR DOMAIN SERVICE (ORCHESTRATOR)
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone Git remote mutator engine implementing atomic, low-level Git plumbing
   primitives:
     (1) Inspect remote branch reference HEAD SHA for Optimistic Concurrency Control (OCC).
     (2) Create remote Git Blobs for modified files via POST /git/blobs.
     (3) Synthesize Git Merkle Tree specifying base_tree and modified/deleted nodes via POST /git/trees.
     (4) Create immutable Git Commit object with author attribution, message, and parent commit links via POST /git/commits.
     (5) Atomically push and update branch reference via CAS PATCH /git/refs/heads/{branch}.
     (6) Synchronize Neo4j knowledge graph (:GitBranch)-[:POINTS_TO]->(:GitCommit).
     (7) Audit log mutation record to AlloyDB SQL ledger.

2. INVARIANTS:
   - ZERO comments inside method bodies (all blueprints declared in this header).
   - Thread-safe, non-blocking Kernel Result wrapping.
   - Comprehensive OpenTelemetry span error instrumentation.
================================================================================
"""

import time
import uuid
from typing import Optional, Dict, Any, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.git_remote_mutator.contracts.git_remote_mutator_deps import (
    GitRemoteMutatorServiceDeps,
)
from src.features.git_remote_mutator.infrastructure.ports.git_remote_mutator_service_port import (
    GitRemoteMutatorServicePort,
)
from src.features.git_remote_mutator.contracts.git_remote_mutator_types import (
    MutateFileCommand,
    MutateBatchCommand,
    DeleteFileCommand,
    FileMutationEntry,
    RemoteBlobCreatedResult,
    RemoteMutationResult,
    EmptyMutationBatchError,
)
from src.features.git_remote_mutator.domain.rules.git_remote_mutator_rules import (
    sanitize_relative_path,
    verify_optimistic_concurrency,
    build_tree_nodes_payload,
)
from src.features.git_remote_mutator.domain.machines.git_remote_mutator_machine import (
    STATE_IDLE,
    STATE_VERIFYING_REF,
    STATE_CREATING_BLOBS,
    STATE_CREATING_TREE,
    STATE_CREATING_COMMIT,
    STATE_UPDATING_REF,
    STATE_UPDATING_GRAPH,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class GitRemoteMutatorDomainService(AbstractTracedComponent, GitRemoteMutatorServicePort):
    def __init__(self, deps: GitRemoteMutatorServiceDeps) -> None:
        super().__init__()
        self._repository = deps.repository
        self._mutation_client = deps.mutation_client
        self._config = deps.config

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def _record_span_error(self, span: Any, exc: Exception) -> None:
        if span:
            if hasattr(span, "record_exception"):
                span.record_exception(exc)
            if StatusCode:
                if Status:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                else:
                    span.set_status(StatusCode.ERROR)

    def mutate_file(
        self, ctx: TraceContext, cmd: MutateFileCommand
    ) -> Result[RemoteMutationResult, Exception]:
        entry = FileMutationEntry(
            path=cmd.file_path,
            content=cmd.content,
            is_deleted=False,
        )
        batch_cmd = MutateBatchCommand(
            repo_url=cmd.repo_url,
            branch=cmd.branch,
            mutations=[entry],
            commit_message=cmd.commit_message,
            expected_parent_sha=cmd.expected_parent_sha,
            author_name=cmd.author_name,
            author_email=cmd.author_email,
            token=cmd.token,
        )
        return self.mutate_batch(ctx=ctx, cmd=batch_cmd)

    def delete_file(
        self, ctx: TraceContext, cmd: DeleteFileCommand
    ) -> Result[RemoteMutationResult, Exception]:
        entry = FileMutationEntry(
            path=cmd.file_path,
            content="",
            is_deleted=True,
        )
        batch_cmd = MutateBatchCommand(
            repo_url=cmd.repo_url,
            branch=cmd.branch,
            mutations=[entry],
            commit_message=cmd.commit_message,
            expected_parent_sha=cmd.expected_parent_sha,
            author_name=cmd.author_name,
            author_email=cmd.author_email,
            token=cmd.token,
        )
        return self.mutate_batch(ctx=ctx, cmd=batch_cmd)

    def mutate_batch(
        self, ctx: TraceContext, cmd: MutateBatchCommand
    ) -> Result[RemoteMutationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_mutator.mutate_batch")

            if not cmd.mutations:
                return Result.err(EmptyMutationBatchError())

            state = transition_to(state, STATE_VERIFYING_REF)
            head_res = self._mutation_client.get_branch_head(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=cmd.branch,
                token=cmd.token,
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            head_data = head_res.unwrap()
            current_head_sha = str(head_data.get("commit_sha", ""))
            verify_optimistic_concurrency(current_head_sha, cmd.expected_parent_sha)

            state = transition_to(state, STATE_CREATING_BLOBS)
            created_blobs: List[RemoteBlobCreatedResult] = []
            blob_shas: Dict[str, str] = {}

            for entry in cmd.mutations:
                clean_path = sanitize_relative_path(entry.path)
                if not entry.is_deleted:
                    blob_res = self._mutation_client.create_blob(
                        ctx=ctx,
                        repo=cmd.repo_url,
                        content=entry.content,
                        encoding="utf-8",
                        token=cmd.token,
                    )
                    if blob_res.is_err:
                        state = transition_to(state, STATE_FAILED)
                        err = blob_res.error()
                        self._record_span_error(span, err)
                        return Result.err(err)
                    b_data = blob_res.unwrap()
                    b_sha = str(b_data.get("sha", ""))
                    blob_shas[clean_path] = b_sha
                    created_blobs.append(
                        RemoteBlobCreatedResult(
                            path=clean_path,
                            sha=b_sha,
                            size=len(entry.content.encode("utf-8")),
                        )
                    )

            state = transition_to(state, STATE_CREATING_TREE)
            tree_nodes = build_tree_nodes_payload(cmd.mutations, blob_shas)
            tree_res = self._mutation_client.create_tree(
                ctx=ctx,
                repo=cmd.repo_url,
                base_tree_sha=current_head_sha,
                tree_nodes=tree_nodes,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            tree_data = tree_res.unwrap()
            new_tree_sha = str(tree_data.get("sha", ""))

            state = transition_to(state, STATE_CREATING_COMMIT)
            commit_res = self._mutation_client.create_commit(
                ctx=ctx,
                repo=cmd.repo_url,
                message=cmd.commit_message,
                tree_sha=new_tree_sha,
                parent_shas=[current_head_sha],
                author_name=cmd.author_name,
                author_email=cmd.author_email,
                token=cmd.token,
            )
            if commit_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = commit_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            commit_data = commit_res.unwrap()
            new_commit_sha = str(commit_data.get("sha", ""))

            state = transition_to(state, STATE_UPDATING_REF)
            ref_res = self._mutation_client.update_ref(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=cmd.branch,
                new_commit_sha=new_commit_sha,
                force=False,
                token=cmd.token,
            )
            if ref_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = ref_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_UPDATING_GRAPH)
            if hasattr(self._repository, "update_branch_pointer"):
                self._repository.update_branch_pointer(
                    ctx=ctx,
                    repo_url=cmd.repo_url,
                    branch=cmd.branch,
                    parent_sha=current_head_sha,
                    new_commit_sha=new_commit_sha,
                    commit_message=cmd.commit_message,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            mutation_record_id = str(uuid.uuid4())
            self._repository.save_mutation_record(
                ctx=ctx,
                record={
                    "id": mutation_record_id,
                    "repo_url": cmd.repo_url,
                    "target_branch": cmd.branch,
                    "parent_commit_sha": current_head_sha,
                    "new_commit_sha": new_commit_sha,
                    "files_count": len(cmd.mutations),
                    "commit_message": cmd.commit_message,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = RemoteMutationResult(
                repo_url=cmd.repo_url,
                branch=cmd.branch,
                parent_commit_sha=current_head_sha,
                new_commit_sha=new_commit_sha,
                tree_sha=new_tree_sha,
                files_mutated_count=len(cmd.mutations),
                commit_message=cmd.commit_message,
                blobs_created=created_blobs,
                duration_ms=duration_ms,
            )

            if span and StatusCode:
                span.set_status(StatusCode.OK)
            return Result.ok(result)
        except Exception as exc:
            state = STATE_FAILED
            self._record_span_error(span, exc)
            return Result.err(exc)
        finally:
            if span:
                span.end()

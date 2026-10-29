"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE DOMAIN SERVICE
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone branch lifecycle domain orchestrator:
     - create_branch: Resolves source ref to SHA, provisions remote branch ref, syncs graph.
     - delete_branch: Deletes remote branch ref, removes node from knowledge graph.
     - merge_branch: Executes Git provider branch merge, records resulting merge commit SHA.
     - create_repository: Provisions brand new remote Git repository without cloning.
     - Persists structured audit ledger records for every lifecycle operation.

2. INVARIANTS:
   - ZERO comments inside method bodies.
   - Comprehensive OpenTelemetry span instrumentation.
   - Returns typed Result[BranchLifecycleResult, Exception].
================================================================================
"""

import time
import uuid
from typing import Optional, Dict, Any
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_deps import (
    GitBranchLifecycleServiceDeps,
)
from src.features.git_branch_lifecycle.infrastructure.ports.git_branch_lifecycle_service_port import (
    GitBranchLifecycleServicePort,
)
from src.features.git_branch_lifecycle.contracts.git_branch_lifecycle_types import (
    CreateBranchCommand,
    DeleteBranchCommand,
    MergeBranchCommand,
    CreateRepositoryCommand,
    BranchLifecycleResult,
)
from src.features.git_branch_lifecycle.domain.rules.git_branch_lifecycle_rules import (
    normalize_branch_name,
    is_sha,
)
from src.features.git_branch_lifecycle.domain.machines.git_branch_lifecycle_machine import (
    STATE_IDLE,
    STATE_VALIDATING_INPUT,
    STATE_EXECUTING_PROVIDER,
    STATE_SYNCING_GRAPH,
    STATE_AUDITING,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class GitBranchLifecycleDomainService(AbstractTracedComponent, GitBranchLifecycleServicePort):
    def __init__(self, deps: GitBranchLifecycleServiceDeps) -> None:
        super().__init__()
        self._repository = deps.repository
        self._client = deps.client
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

    def create_branch(
        self, ctx: TraceContext, cmd: CreateBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_branch_lifecycle.create_branch")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_branch = normalize_branch_name(cmd.branch_name)

            state = transition_to(state, STATE_EXECUTING_PROVIDER)
            target_sha = cmd.source_ref.strip()
            if not is_sha(target_sha):
                res_sha = self._client.resolve_ref_sha(
                    ctx=ctx, repo=cmd.repo_url, ref=cmd.source_ref, token=cmd.token
                )
                if res_sha.is_err:
                    state = transition_to(state, STATE_FAILED)
                    self._record_span_error(span, res_sha.error())
                    return Result.err(res_sha.error())
                target_sha = res_sha.unwrap()

            create_res = self._client.create_branch_ref(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=clean_branch,
                target_sha=target_sha,
                token=cmd.token,
            )
            if create_res.is_err:
                state = transition_to(state, STATE_FAILED)
                self._record_span_error(span, create_res.error())
                return Result.err(create_res.error())

            state = transition_to(state, STATE_SYNCING_GRAPH)
            self._repository.create_branch_node(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=clean_branch,
                commit_sha=target_sha,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_AUDITING)
            self._repository.save_lifecycle_record(
                ctx=ctx,
                record={
                    "id": str(uuid.uuid4()),
                    "repo_url": cmd.repo_url,
                    "action": "CREATE_BRANCH",
                    "source_ref": cmd.source_ref,
                    "target_ref": clean_branch,
                    "result_sha": target_sha,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = BranchLifecycleResult(
                repo_url=cmd.repo_url,
                action="CREATE_BRANCH",
                branch_name=clean_branch,
                source_ref=cmd.source_ref,
                target_ref=clean_branch,
                result_sha=target_sha,
                is_success=True,
                execution_ms=duration_ms,
                details=create_res.unwrap(),
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

    def delete_branch(
        self, ctx: TraceContext, cmd: DeleteBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_branch_lifecycle.delete_branch")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_branch = normalize_branch_name(cmd.branch_name)

            state = transition_to(state, STATE_EXECUTING_PROVIDER)
            del_res = self._client.delete_branch_ref(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=clean_branch,
                token=cmd.token,
            )
            if del_res.is_err:
                state = transition_to(state, STATE_FAILED)
                self._record_span_error(span, del_res.error())
                return Result.err(del_res.error())

            state = transition_to(state, STATE_SYNCING_GRAPH)
            self._repository.delete_branch_node(
                ctx=ctx,
                repo=cmd.repo_url,
                branch=clean_branch,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_AUDITING)
            self._repository.save_lifecycle_record(
                ctx=ctx,
                record={
                    "id": str(uuid.uuid4()),
                    "repo_url": cmd.repo_url,
                    "action": "DELETE_BRANCH",
                    "source_ref": clean_branch,
                    "target_ref": None,
                    "result_sha": None,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = BranchLifecycleResult(
                repo_url=cmd.repo_url,
                action="DELETE_BRANCH",
                branch_name=clean_branch,
                source_ref=clean_branch,
                target_ref=None,
                result_sha=None,
                is_success=True,
                execution_ms=duration_ms,
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

    def merge_branch(
        self, ctx: TraceContext, cmd: MergeBranchCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_branch_lifecycle.merge_branch")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_base = normalize_branch_name(cmd.base_branch)
            clean_head = normalize_branch_name(cmd.head_branch)

            state = transition_to(state, STATE_EXECUTING_PROVIDER)
            merge_res = self._client.merge_branches(
                ctx=ctx,
                repo=cmd.repo_url,
                base=clean_base,
                head=clean_head,
                commit_message=cmd.commit_message,
                token=cmd.token,
            )
            if merge_res.is_err:
                state = transition_to(state, STATE_FAILED)
                self._record_span_error(span, merge_res.error())
                return Result.err(merge_res.error())

            merge_data = merge_res.unwrap()
            merge_sha = str(merge_data.get("sha", ""))

            state = transition_to(state, STATE_SYNCING_GRAPH)
            if merge_sha:
                self._repository.create_branch_node(
                    ctx=ctx,
                    repo=cmd.repo_url,
                    branch=clean_base,
                    commit_sha=merge_sha,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_AUDITING)
            self._repository.save_lifecycle_record(
                ctx=ctx,
                record={
                    "id": str(uuid.uuid4()),
                    "repo_url": cmd.repo_url,
                    "action": "MERGE_BRANCH",
                    "source_ref": clean_head,
                    "target_ref": clean_base,
                    "result_sha": merge_sha,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = BranchLifecycleResult(
                repo_url=cmd.repo_url,
                action="MERGE_BRANCH",
                branch_name=clean_base,
                source_ref=clean_head,
                target_ref=clean_base,
                result_sha=merge_sha,
                is_success=True,
                execution_ms=duration_ms,
                details=merge_data,
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

    def create_repository(
        self, ctx: TraceContext, cmd: CreateRepositoryCommand
    ) -> Result[BranchLifecycleResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_branch_lifecycle.create_repository")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_name = cmd.repo_name.strip()
            if not clean_name:
                raise ValueError("Repository name cannot be empty")

            state = transition_to(state, STATE_EXECUTING_PROVIDER)
            repo_res = self._client.create_remote_repo(
                ctx=ctx,
                name=clean_name,
                org=cmd.org,
                description=cmd.description,
                is_private=cmd.is_private,
                auto_init=cmd.auto_init,
                token=cmd.token,
            )
            if repo_res.is_err:
                state = transition_to(state, STATE_FAILED)
                self._record_span_error(span, repo_res.error())
                return Result.err(repo_res.error())

            repo_data = repo_res.unwrap()
            full_repo_url = str(repo_data.get("html_url", f"https://github.com/{clean_name}"))

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_AUDITING)
            self._repository.save_lifecycle_record(
                ctx=ctx,
                record={
                    "id": str(uuid.uuid4()),
                    "repo_url": full_repo_url,
                    "action": "CREATE_REPO",
                    "source_ref": None,
                    "target_ref": "main",
                    "result_sha": None,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = BranchLifecycleResult(
                repo_url=full_repo_url,
                action="CREATE_REPO",
                branch_name="main",
                source_ref=None,
                target_ref="main",
                result_sha=None,
                is_success=True,
                execution_ms=duration_ms,
                details=repo_data,
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

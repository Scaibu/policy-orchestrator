"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER DOMAIN SERVICE (HEXAGONAL ORCHESTRATOR)
================================================================================
1. OVERVIEW & OBJECTIVE:
   Zero-clone multi-repo Git submodule domain orchestrator:
     - `add_submodule`: registers child repository URL and path in .gitmodules,
       resolves child HEAD commit, stages mode 160000 gitlink, and commits.
     - `update_submodule`: updates a single submodule gitlink pointer to target SHA or branch HEAD.
     - `pull_submodules`: batches update across registered submodules pulling latest upstream commits.
     - `push_multi_repo`: ensures child commits exist on remote before parent commit is published.
     - `get_sync_status`: audits parent tree gitlinks vs child remote branch HEADs and computes drift.
     - Persists structured audit records and updates Neo4j submodule graph.

2. INVARIANTS:
   - ZERO comments inside method bodies (all blueprints declared in this header).
   - Thread-safe, non-blocking Kernel Result wrapping.
   - Comprehensive OpenTelemetry span error instrumentation.
   - Strictly conforms to 4-functions-and-classes-writing-rule.md (try/catch/finally).
================================================================================
"""

import time
import uuid
from typing import Optional, Dict, Any, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.git_submodule_manager.contracts.git_submodule_manager_deps import (
    GitSubmoduleManagerServiceDeps,
)
from src.features.git_submodule_manager.infrastructure.ports.git_submodule_manager_service_port import (
    GitSubmoduleManagerServicePort,
)
from src.features.git_submodule_manager.contracts.git_submodule_manager_types import (
    AddSubmoduleCommand,
    UpdateSubmoduleCommand,
    PullSubmodulesCommand,
    PushMultiRepoCommand,
    SubmoduleSyncStatusCommand,
    SubmoduleOperationResult,
    SubmoduleItemStatus,
    SubmoduleNotFoundError,
    SubmoduleRemoteCommitMissingError,
    InvalidSubmoduleSpecifierError,
)
from src.features.git_submodule_manager.domain.rules.git_submodule_manager_rules import (
    sanitize_submodule_path,
    parse_gitmodules_content,
    upsert_submodule_in_gitmodules,
    build_gitlink_tree_nodes,
    is_valid_sha,
    calculate_commit_drift,
)
from src.features.git_submodule_manager.domain.machines.git_submodule_manager_machine import (
    STATE_IDLE,
    STATE_VALIDATING_INPUT,
    STATE_INSPECTING_CHILD_REMOTES,
    STATE_PREPARING_TREE_MUTATION,
    STATE_CREATING_COMMIT,
    STATE_UPDATING_REF,
    STATE_SYNCING_GRAPH,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)
from src.features.git_remote_mutator.domain.rules.git_remote_mutator_rules import (
    verify_optimistic_concurrency,
)

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class GitSubmoduleManagerDomainService(AbstractTracedComponent, GitSubmoduleManagerServicePort):
    def __init__(self, deps: GitSubmoduleManagerServiceDeps) -> None:
        super().__init__()
        self._repository = deps.repository
        self._mutator_client = deps.mutator_client
        self._inspector_client = deps.inspector_client
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

    def _fetch_gitmodules_content(
        self, ctx: TraceContext, repo: str, ref: str, token: Optional[str]
    ) -> str:
        res = self._inspector_client.fetch_file_content(ctx, repo, ".gitmodules", ref, token)
        if res.is_ok:
            data = res.unwrap()
            return str(data.get("content", ""))
        return ""

    def add_submodule(
        self, ctx: TraceContext, cmd: AddSubmoduleCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_submodule_manager.add_submodule")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_path = sanitize_submodule_path(cmd.submodule_path)

            head_res = self._mutator_client.get_branch_head(
                ctx=ctx, repo=cmd.repo_url, branch=cmd.branch, token=cmd.token
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            parent_head_data = head_res.unwrap()
            current_parent_head_sha = str(parent_head_data.get("commit_sha", ""))
            verify_optimistic_concurrency(current_parent_head_sha, cmd.expected_parent_sha)

            state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
            target_child_sha = cmd.submodule_ref
            if not is_valid_sha(target_child_sha):
                child_head_res = self._mutator_client.get_branch_head(
                    ctx=ctx, repo=cmd.submodule_url, branch=cmd.submodule_ref, token=cmd.token
                )
                if child_head_res.is_err:
                    state = transition_to(state, STATE_FAILED)
                    err = child_head_res.error()
                    self._record_span_error(span, err)
                    return Result.err(err)
                target_child_sha = str(child_head_res.unwrap().get("commit_sha", ""))

            existing_gitmodules = self._fetch_gitmodules_content(
                ctx, cmd.repo_url, current_parent_head_sha, cmd.token
            )
            updated_gitmodules = upsert_submodule_in_gitmodules(
                existing_gitmodules, clean_path, cmd.submodule_url, cmd.submodule_ref
            )

            state = transition_to(state, STATE_PREPARING_TREE_MUTATION)
            blob_res = self._mutator_client.create_blob(
                ctx=ctx, repo=cmd.repo_url, content=updated_gitmodules, encoding="utf-8", token=cmd.token
            )
            if blob_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = blob_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            gitmodules_blob_sha = str(blob_res.unwrap().get("sha", ""))

            tree_nodes = build_gitlink_tree_nodes(
                submodule_paths_to_shas={clean_path: target_child_sha},
                gitmodules_blob_sha=gitmodules_blob_sha,
            )

            tree_res = self._mutator_client.create_tree(
                ctx=ctx,
                repo=cmd.repo_url,
                base_tree_sha=current_parent_head_sha,
                tree_nodes=tree_nodes,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_tree_sha = str(tree_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_CREATING_COMMIT)
            commit_res = self._mutator_client.create_commit(
                ctx=ctx,
                repo=cmd.repo_url,
                message=cmd.commit_message,
                tree_sha=new_tree_sha,
                parent_shas=[current_parent_head_sha],
                author_name=cmd.author_name,
                author_email=cmd.author_email,
                token=cmd.token,
            )
            if commit_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = commit_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_commit_sha = str(commit_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_UPDATING_REF)
            ref_res = self._mutator_client.update_ref(
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

            state = transition_to(state, STATE_SYNCING_GRAPH)
            self._repository.sync_submodule_graph(
                ctx=ctx,
                parent_repo=cmd.repo_url,
                submodule_path=clean_path,
                submodule_url=cmd.submodule_url,
                gitlink_sha=target_child_sha,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            rec_id = str(uuid.uuid4())
            self._repository.save_submodule_record(
                ctx=ctx,
                record={
                    "id": rec_id,
                    "repo_url": cmd.repo_url,
                    "branch": cmd.branch,
                    "action": "ADD_SUBMODULE",
                    "submodule_path": clean_path,
                    "submodule_url": cmd.submodule_url,
                    "gitlink_sha": target_child_sha,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = SubmoduleOperationResult(
                repo_url=cmd.repo_url,
                branch=cmd.branch,
                action="ADD_SUBMODULE",
                parent_commit_sha=current_parent_head_sha,
                new_commit_sha=new_commit_sha,
                tree_sha=new_tree_sha,
                submodules_affected=[clean_path],
                commit_message=cmd.commit_message,
                is_noop=False,
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

    def update_submodule(
        self, ctx: TraceContext, cmd: UpdateSubmoduleCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_submodule_manager.update_submodule")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            clean_path = sanitize_submodule_path(cmd.submodule_path)

            head_res = self._mutator_client.get_branch_head(
                ctx=ctx, repo=cmd.repo_url, branch=cmd.branch, token=cmd.token
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            current_parent_head_sha = str(head_res.unwrap().get("commit_sha", ""))
            verify_optimistic_concurrency(current_parent_head_sha, cmd.expected_parent_sha)

            gitmodules_content = self._fetch_gitmodules_content(
                ctx, cmd.repo_url, current_parent_head_sha, cmd.token
            )
            modules = parse_gitmodules_content(gitmodules_content)
            if clean_path not in modules:
                state = transition_to(state, STATE_FAILED)
                err = SubmoduleNotFoundError(f"Submodule path '{clean_path}' not declared in .gitmodules")
                self._record_span_error(span, err)
                return Result.err(err)

            sub_info = modules[clean_path]
            sub_url = sub_info.get("url", "")
            target_sha = cmd.target_commit_sha

            state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
            if not target_sha:
                target_branch = cmd.submodule_branch or sub_info.get("branch", "main")
                child_head_res = self._mutator_client.get_branch_head(
                    ctx=ctx, repo=sub_url, branch=target_branch, token=cmd.token
                )
                if child_head_res.is_err:
                    state = transition_to(state, STATE_FAILED)
                    err = child_head_res.error()
                    self._record_span_error(span, err)
                    return Result.err(err)
                target_sha = str(child_head_res.unwrap().get("commit_sha", ""))

            state = transition_to(state, STATE_PREPARING_TREE_MUTATION)
            tree_nodes = build_gitlink_tree_nodes(submodule_paths_to_shas={clean_path: target_sha})
            tree_res = self._mutator_client.create_tree(
                ctx=ctx,
                repo=cmd.repo_url,
                base_tree_sha=current_parent_head_sha,
                tree_nodes=tree_nodes,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_tree_sha = str(tree_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_CREATING_COMMIT)
            commit_res = self._mutator_client.create_commit(
                ctx=ctx,
                repo=cmd.repo_url,
                message=cmd.commit_message,
                tree_sha=new_tree_sha,
                parent_shas=[current_parent_head_sha],
                author_name=cmd.author_name,
                author_email=cmd.author_email,
                token=cmd.token,
            )
            if commit_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = commit_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_commit_sha = str(commit_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_UPDATING_REF)
            ref_res = self._mutator_client.update_ref(
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

            state = transition_to(state, STATE_SYNCING_GRAPH)
            self._repository.sync_submodule_graph(
                ctx=ctx,
                parent_repo=cmd.repo_url,
                submodule_path=clean_path,
                submodule_url=sub_url,
                gitlink_sha=target_sha,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            rec_id = str(uuid.uuid4())
            self._repository.save_submodule_record(
                ctx=ctx,
                record={
                    "id": rec_id,
                    "repo_url": cmd.repo_url,
                    "branch": cmd.branch,
                    "action": "UPDATE_SUBMODULE",
                    "submodule_path": clean_path,
                    "submodule_url": sub_url,
                    "gitlink_sha": target_sha,
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = SubmoduleOperationResult(
                repo_url=cmd.repo_url,
                branch=cmd.branch,
                action="UPDATE_SUBMODULE",
                parent_commit_sha=current_parent_head_sha,
                new_commit_sha=new_commit_sha,
                tree_sha=new_tree_sha,
                submodules_affected=[clean_path],
                commit_message=cmd.commit_message,
                is_noop=False,
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

    def pull_submodules(
        self, ctx: TraceContext, cmd: PullSubmodulesCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_submodule_manager.pull_submodules")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            head_res = self._mutator_client.get_branch_head(
                ctx=ctx, repo=cmd.repo_url, branch=cmd.branch, token=cmd.token
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            current_parent_head_sha = str(head_res.unwrap().get("commit_sha", ""))
            verify_optimistic_concurrency(current_parent_head_sha, cmd.expected_parent_sha)

            gitmodules_content = self._fetch_gitmodules_content(
                ctx, cmd.repo_url, current_parent_head_sha, cmd.token
            )
            modules = parse_gitmodules_content(gitmodules_content)

            targets = cmd.submodule_paths or list(modules.keys())
            state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
            updates: Dict[str, str] = {}
            for path in targets:
                clean_path = sanitize_submodule_path(path)
                if clean_path in modules:
                    sub_info = modules[clean_path]
                    sub_url = sub_info.get("url", "")
                    sub_branch = sub_info.get("branch", "main")
                    child_head_res = self._mutator_client.get_branch_head(
                        ctx=ctx, repo=sub_url, branch=sub_branch, token=cmd.token
                    )
                    if child_head_res.is_ok:
                        updates[clean_path] = str(child_head_res.unwrap().get("commit_sha", ""))

            state = transition_to(state, STATE_PREPARING_TREE_MUTATION)
            tree_nodes = build_gitlink_tree_nodes(submodule_paths_to_shas=updates)
            tree_res = self._mutator_client.create_tree(
                ctx=ctx,
                repo=cmd.repo_url,
                base_tree_sha=current_parent_head_sha,
                tree_nodes=tree_nodes,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_tree_sha = str(tree_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_CREATING_COMMIT)
            commit_res = self._mutator_client.create_commit(
                ctx=ctx,
                repo=cmd.repo_url,
                message=cmd.commit_message,
                tree_sha=new_tree_sha,
                parent_shas=[current_parent_head_sha],
                author_name=cmd.author_name,
                author_email=cmd.author_email,
                token=cmd.token,
            )
            if commit_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = commit_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_commit_sha = str(commit_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_UPDATING_REF)
            ref_res = self._mutator_client.update_ref(
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

            state = transition_to(state, STATE_SYNCING_GRAPH)
            for p, sha in updates.items():
                self._repository.sync_submodule_graph(
                    ctx=ctx,
                    parent_repo=cmd.repo_url,
                    submodule_path=p,
                    submodule_url=modules[p].get("url", ""),
                    gitlink_sha=sha,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            rec_id = str(uuid.uuid4())
            self._repository.save_submodule_record(
                ctx=ctx,
                record={
                    "id": rec_id,
                    "repo_url": cmd.repo_url,
                    "branch": cmd.branch,
                    "action": "PULL_SUBMODULES",
                    "submodule_path": ",".join(updates.keys()),
                    "submodule_url": "",
                    "gitlink_sha": "",
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = SubmoduleOperationResult(
                repo_url=cmd.repo_url,
                branch=cmd.branch,
                action="PULL_SUBMODULES",
                parent_commit_sha=current_parent_head_sha,
                new_commit_sha=new_commit_sha,
                tree_sha=new_tree_sha,
                submodules_affected=list(updates.keys()),
                commit_message=cmd.commit_message,
                is_noop=False,
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

    def push_multi_repo(
        self, ctx: TraceContext, cmd: PushMultiRepoCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_submodule_manager.push_multi_repo")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            head_res = self._mutator_client.get_branch_head(
                ctx=ctx, repo=cmd.parent_repo_url, branch=cmd.parent_branch, token=cmd.token
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            current_parent_head_sha = str(head_res.unwrap().get("commit_sha", ""))
            verify_optimistic_concurrency(current_parent_head_sha, cmd.expected_parent_sha)

            gitmodules_content = self._fetch_gitmodules_content(
                ctx, cmd.parent_repo_url, current_parent_head_sha, cmd.token
            )
            modules = parse_gitmodules_content(gitmodules_content)

            state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
            verified_submodules: Dict[str, str] = {}
            target_paths = cmd.submodule_paths if cmd.submodule_paths is not None else list(modules.keys())
            for path in target_paths:
                clean_path = sanitize_submodule_path(path)
                if clean_path not in modules:
                    state = transition_to(state, STATE_FAILED)
                    err = SubmoduleNotFoundError(f"Submodule path '{clean_path}' not declared in .gitmodules")
                    self._record_span_error(span, err)
                    return Result.err(err)
                sub_info = modules[clean_path]
                sub_url = sub_info.get("url", "")
                sub_branch = sub_info.get("branch", "main")

                child_head_res = self._mutator_client.get_branch_head(
                    ctx=ctx, repo=sub_url, branch=sub_branch, token=cmd.token
                )
                if child_head_res.is_err:
                    state = transition_to(state, STATE_FAILED)
                    err = child_head_res.error()
                    self._record_span_error(span, err)
                    return Result.err(err)
                child_sha = str(child_head_res.unwrap().get("commit_sha", ""))
                if cmd.verify_remote_exists and not child_sha:
                    state = transition_to(state, STATE_FAILED)
                    err = SubmoduleRemoteCommitMissingError(
                        f"Submodule '{clean_path}' commit cannot be verified on remote repository '{sub_url}'"
                    )
                    self._record_span_error(span, err)
                    return Result.err(err)
                verified_submodules[clean_path] = child_sha

            state = transition_to(state, STATE_PREPARING_TREE_MUTATION)
            tree_nodes = build_gitlink_tree_nodes(submodule_paths_to_shas=verified_submodules)
            tree_res = self._mutator_client.create_tree(
                ctx=ctx,
                repo=cmd.parent_repo_url,
                base_tree_sha=current_parent_head_sha,
                tree_nodes=tree_nodes,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_tree_sha = str(tree_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_CREATING_COMMIT)
            commit_res = self._mutator_client.create_commit(
                ctx=ctx,
                repo=cmd.parent_repo_url,
                message=cmd.commit_message,
                tree_sha=new_tree_sha,
                parent_shas=[current_parent_head_sha],
                author_name=cmd.author_name,
                author_email=cmd.author_email,
                token=cmd.token,
            )
            if commit_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = commit_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            new_commit_sha = str(commit_res.unwrap().get("sha", ""))

            state = transition_to(state, STATE_UPDATING_REF)
            ref_res = self._mutator_client.update_ref(
                ctx=ctx,
                repo=cmd.parent_repo_url,
                branch=cmd.parent_branch,
                new_commit_sha=new_commit_sha,
                force=False,
                token=cmd.token,
            )
            if ref_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = ref_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_SYNCING_GRAPH)
            for p, sha in verified_submodules.items():
                self._repository.sync_submodule_graph(
                    ctx=ctx,
                    parent_repo=cmd.parent_repo_url,
                    submodule_path=p,
                    submodule_url=modules[p].get("url", ""),
                    gitlink_sha=sha,
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            rec_id = str(uuid.uuid4())
            self._repository.save_submodule_record(
                ctx=ctx,
                record={
                    "id": rec_id,
                    "repo_url": cmd.parent_repo_url,
                    "branch": cmd.parent_branch,
                    "action": "PUSH_MULTI_REPO",
                    "submodule_path": ",".join(verified_submodules.keys()),
                    "submodule_url": "",
                    "gitlink_sha": "",
                    "is_success": True,
                    "execution_ms": duration_ms,
                },
            )

            state = transition_to(state, STATE_COMPLETED)
            result = SubmoduleOperationResult(
                repo_url=cmd.parent_repo_url,
                branch=cmd.parent_branch,
                action="PUSH_MULTI_REPO",
                parent_commit_sha=current_parent_head_sha,
                new_commit_sha=new_commit_sha,
                tree_sha=new_tree_sha,
                submodules_affected=list(verified_submodules.keys()),
                commit_message=cmd.commit_message,
                is_noop=False,
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

    def get_sync_status(
        self, ctx: TraceContext, cmd: SubmoduleSyncStatusCommand
    ) -> Result[SubmoduleOperationResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_submodule_manager.get_sync_status")

            state = transition_to(state, STATE_VALIDATING_INPUT)
            head_res = self._mutator_client.get_branch_head(
                ctx=ctx, repo=cmd.repo_url, branch=cmd.branch, token=cmd.token
            )
            if head_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = head_res.error()
                self._record_span_error(span, err)
                return Result.err(err)
            current_parent_head_sha = str(head_res.unwrap().get("commit_sha", ""))

            gitmodules_content = self._fetch_gitmodules_content(
                ctx, cmd.repo_url, current_parent_head_sha, cmd.token
            )
            modules = parse_gitmodules_content(gitmodules_content)

            state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
            status_items: List[SubmoduleItemStatus] = []
            for path, info in modules.items():
                clean_path = sanitize_submodule_path(path)
                sub_url = info.get("url", "")
                sub_branch = info.get("branch", "main")

                child_head_res = self._mutator_client.get_branch_head(
                    ctx=ctx, repo=sub_url, branch=sub_branch, token=cmd.token
                )
                remote_sha = str(child_head_res.unwrap().get("commit_sha", "")) if child_head_res.is_ok else ""

                tree_res = self._inspector_client.fetch_tree(
                    ctx=ctx, repo=cmd.repo_url, ref=current_parent_head_sha, token=cmd.token
                )
                gitlink_sha = ""
                if tree_res.is_ok:
                    tree_data = tree_res.unwrap()
                    for item in tree_data.get("tree", []):
                        if item.get("path") == clean_path and item.get("mode") == "160000":
                            gitlink_sha = str(item.get("sha", ""))
                            break

                in_sync, drift = calculate_commit_drift(gitlink_sha, remote_sha)
                status_items.append(
                    SubmoduleItemStatus(
                        name=info.get("name", clean_path),
                        path=clean_path,
                        url=sub_url,
                        branch=sub_branch,
                        gitlink_sha=gitlink_sha,
                        remote_head_sha=remote_sha,
                        is_in_sync=in_sync,
                        drift_commit_count=drift,
                    )
                )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_COMPLETED)
            result = SubmoduleOperationResult(
                repo_url=cmd.repo_url,
                branch=cmd.branch,
                action="SYNC_STATUS",
                parent_commit_sha=current_parent_head_sha,
                new_commit_sha=current_parent_head_sha,
                tree_sha="",
                submodules_affected=[s.path for s in status_items],
                commit_message="Status check",
                is_noop=True,
                duration_ms=duration_ms,
                status_details=status_items,
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

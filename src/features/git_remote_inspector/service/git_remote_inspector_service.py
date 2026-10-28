import time
import uuid
from typing import Optional, Dict, Any, List
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.features.git_remote_inspector.contracts.git_remote_inspector_deps import (
    GitRemoteInspectorServiceDeps,
)
from src.features.git_remote_inspector.infrastructure.ports.git_remote_inspector_service_port import (
    GitRemoteInspectorServicePort,
)
from src.features.git_remote_inspector.contracts.git_remote_inspector_types import (
    SearchFilesCommand,
    GetFileContentCommand,
    GetDiffCommand,
    RemoteInspectionResult,
    RemoteFileContentResult,
    RemoteDiffResult,
    RemoteTreeItem,
)
from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
    filter_tree_nodes,
    sanitize_file_path,
    extract_diff_patch,
)
from src.features.git_remote_inspector.domain.machines.git_remote_inspector_machine import (
    STATE_IDLE,
    STATE_FETCHING_TREE,
    STATE_FILTERING_NODES,
    STATE_FETCHING_CONTENT,
    STATE_COMPARING_DIFF,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class GitRemoteInspectorDomainService(AbstractTracedComponent, GitRemoteInspectorServicePort):
    def __init__(self, deps: GitRemoteInspectorServiceDeps) -> None:
        super().__init__()
        self._repository = deps.repository
        self._remote_client = deps.remote_client
        self._config = deps.config

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def search_files(
        self, ctx: TraceContext, cmd: SearchFilesCommand
    ) -> Result[RemoteInspectionResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.search_files")
            state = transition_to(state, STATE_FETCHING_TREE)
            tree_res = self._remote_client.fetch_tree(
                ctx=ctx,
                repo=cmd.repo_url,
                ref=cmd.ref_name,
                recursive=True,
                token=cmd.token,
            )
            if tree_res.is_err:
                state = transition_to(state, STATE_FAILED)
                return Result.err(tree_res.error())

            state = transition_to(state, STATE_FILTERING_NODES)
            tree_data = tree_res.unwrap()
            commit_sha = str(tree_data.get("sha", cmd.ref_name))
            raw_tree = tree_data.get("tree", [])
            filtered_nodes: List[RemoteTreeItem] = filter_tree_nodes(raw_tree, cmd.path_pattern)

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteInspectionResult(
                repo_url=cmd.repo_url,
                ref_name=cmd.ref_name,
                commit_sha=commit_sha,
                total_found=len(filtered_nodes),
                files=filtered_nodes,
                duration_ms=duration_ms,
            )

            record_id = str(uuid.uuid4())
            self._repository.save_inspection_record(
                ctx,
                {
                    "id": record_id,
                    "repo_url": cmd.repo_url,
                    "ref_name": cmd.ref_name,
                    "path_pattern": cmd.path_pattern,
                    "total_found": len(filtered_nodes),
                    "execution_ms": duration_ms,
                },
            )

            if span and StatusCode:
                span.set_status(StatusCode.OK)
            return Result.ok(result)
        except Exception as exc:
            state = STATE_FAILED
            if span:
                if hasattr(span, "record_exception"):
                    span.record_exception(exc)
                if StatusCode:
                    span.set_status(StatusCode.ERROR)
            return Result.err(exc)
        finally:
            if span:
                span.end()

    def get_file_content(
        self, ctx: TraceContext, cmd: GetFileContentCommand
    ) -> Result[RemoteFileContentResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.get_file_content")
            state = transition_to(state, STATE_FETCHING_CONTENT)
            safe_path = sanitize_file_path(cmd.file_path)
            content_res = self._remote_client.fetch_file_content(
                ctx=ctx,
                repo=cmd.repo_url,
                path=safe_path,
                ref=cmd.ref_name,
                token=cmd.token,
            )
            if content_res.is_err:
                state = transition_to(state, STATE_FAILED)
                return Result.err(content_res.error())

            payload = content_res.unwrap()
            blob_sha = str(payload.get("sha", ""))
            content_str = str(payload.get("decoded_content", payload.get("content", "")))
            size_bytes = int(payload.get("size", len(content_str.encode("utf-8"))))

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteFileContentResult(
                repo_url=cmd.repo_url,
                ref_name=cmd.ref_name,
                file_path=safe_path,
                blob_sha=blob_sha,
                content=content_str,
                encoding="utf-8",
                size_bytes=size_bytes,
                duration_ms=duration_ms,
            )
            if span and StatusCode:
                span.set_status(StatusCode.OK)
            return Result.ok(result)
        except Exception as exc:
            state = STATE_FAILED
            if span:
                if hasattr(span, "record_exception"):
                    span.record_exception(exc)
                if StatusCode:
                    span.set_status(StatusCode.ERROR)
            return Result.err(exc)
        finally:
            if span:
                span.end()

    def get_diff(
        self, ctx: TraceContext, cmd: GetDiffCommand
    ) -> Result[RemoteDiffResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.get_diff")
            state = transition_to(state, STATE_COMPARING_DIFF)
            diff_res = self._remote_client.fetch_diff(
                ctx=ctx,
                repo=cmd.repo_url,
                base_ref=cmd.base_ref,
                head_ref=cmd.head_ref,
                token=cmd.token,
            )
            if diff_res.is_err:
                state = transition_to(state, STATE_FAILED)
                return Result.err(diff_res.error())

            payload = diff_res.unwrap()
            files_changed, patch_str = extract_diff_patch(payload)

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteDiffResult(
                repo_url=cmd.repo_url,
                base_ref=cmd.base_ref,
                head_ref=cmd.head_ref,
                files_changed_count=files_changed,
                diff_patch=patch_str,
                duration_ms=duration_ms,
            )
            if span and StatusCode:
                span.set_status(StatusCode.OK)
            return Result.ok(result)
        except Exception as exc:
            state = STATE_FAILED
            if span:
                if hasattr(span, "record_exception"):
                    span.record_exception(exc)
                if StatusCode:
                    span.set_status(StatusCode.ERROR)
            return Result.err(exc)
        finally:
            if span:
                span.end()

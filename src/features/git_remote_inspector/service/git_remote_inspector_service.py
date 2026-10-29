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
    GetTreeHierarchyCommand,
    ListBranchesCommand,
    DiscoverContractsCommand,
    DiscoveredContractItem,
    TraceCallChainCommand,
    CallChainNode,
    RemoteInspectionResult,
    RemoteFileContentResult,
    RemoteDiffResult,
    RemoteFolderHierarchyResult,
    RemoteBranchesResult,
    RemoteDiscoveredContractsResult,
    RemoteCallChainResult,
    RemoteBranchItem,
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
    STATE_FETCHING_BRANCHES,
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

    def _record_span_error(self, span: Any, exc: Exception) -> None:
        if span:
            if hasattr(span, "record_exception"):
                span.record_exception(exc)
            if StatusCode:
                if Status:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                else:
                    span.set_status(StatusCode.ERROR)

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
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_FILTERING_NODES)
            tree_data = tree_res.unwrap()
            commit_sha = str(tree_data.get("sha", cmd.ref_name))
            raw_tree = tree_data.get("tree", [])
            truncated_flag = bool(tree_data.get("truncated", False))
            provider_name = str(tree_data.get("provider", "github"))
            filtered_nodes: List[RemoteTreeItem] = filter_tree_nodes(raw_tree, cmd.path_pattern)

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteInspectionResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
                ref_name=cmd.ref_name,
                commit_sha=commit_sha,
                total_found=len(filtered_nodes),
                truncated=truncated_flag,
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
            self._record_span_error(span, exc)
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
                err = content_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            payload = content_res.unwrap()
            blob_sha = str(payload.get("sha", ""))
            content_str = str(payload.get("decoded_content", payload.get("content", "")))
            size_bytes = int(payload.get("size", len(content_str.encode("utf-8"))))
            provider_name = str(payload.get("provider", "github"))

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteFileContentResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
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
            self._record_span_error(span, exc)
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
                err = diff_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            payload = diff_res.unwrap()
            files_changed, patch_str = extract_diff_patch(payload)
            provider_name = str(payload.get("provider", "github"))

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteDiffResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
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
            self._record_span_error(span, exc)
            return Result.err(exc)
        finally:
            if span:
                span.end()

    def get_tree_hierarchy(
        self, ctx: TraceContext, cmd: GetTreeHierarchyCommand
    ) -> Result[RemoteFolderHierarchyResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.get_tree_hierarchy")
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
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_FILTERING_NODES)
            tree_data = tree_res.unwrap()
            commit_sha = str(tree_data.get("sha", cmd.ref_name))
            raw_tree = tree_data.get("tree", [])
            truncated_flag = bool(tree_data.get("truncated", False))
            provider_name = str(tree_data.get("provider", "github"))

            from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
                build_nested_tree_hierarchy,
            )
            tree_hierarchy, file_count, dir_count = build_nested_tree_hierarchy(raw_tree, cmd.max_depth)

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteFolderHierarchyResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
                ref_name=cmd.ref_name,
                commit_sha=commit_sha,
                total_files=file_count,
                total_directories=dir_count,
                truncated=truncated_flag,
                tree_hierarchy=tree_hierarchy,
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

    def list_branches(
        self, ctx: TraceContext, cmd: ListBranchesCommand
    ) -> Result[RemoteBranchesResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.list_branches")
            state = transition_to(state, STATE_FETCHING_BRANCHES)
            branches_res = self._remote_client.fetch_branches(
                ctx=ctx,
                repo=cmd.repo_url,
                token=cmd.token,
            )
            if branches_res.is_err:
                state = transition_to(state, STATE_FAILED)
                err = branches_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            data = branches_res.unwrap()
            raw_branches = data.get("branches", [])
            items = [
                RemoteBranchItem(
                    name=str(b.get("name", "")),
                    commit_sha=str(b.get("commit_sha", "")),
                    is_default=bool(b.get("is_default", False)),
                    protected=bool(b.get("protected", False)),
                )
                for b in raw_branches
            ]
            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteBranchesResult(
                repo_url=cmd.repo_url,
                provider=str(data.get("provider", "github")),
                default_branch=str(data.get("default_branch", "main")),
                total_branches=len(items),
                branches=items,
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

    def discover_contracts(
        self, ctx: TraceContext, cmd: DiscoverContractsCommand
    ) -> Result[RemoteDiscoveredContractsResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.discover_contracts")
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
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_FILTERING_NODES)
            tree_data = tree_res.unwrap()
            commit_sha = str(tree_data.get("sha", cmd.ref_name))
            raw_tree = tree_data.get("tree", [])
            provider_name = str(tree_data.get("provider", "github"))

            from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
                categorize_contract_nodes,
            )
            primary_lang, discovered_nodes = categorize_contract_nodes(raw_tree)

            contract_items = [
                DiscoveredContractItem(
                    category=item["category"],
                    priority=item["priority"],
                    path=item["path"],
                    sha=item["sha"],
                    size=item["size"],
                )
                for item in discovered_nodes
            ]

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteDiscoveredContractsResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
                ref_name=cmd.ref_name,
                commit_sha=commit_sha,
                primary_language=primary_lang,
                total_contracts=len(contract_items),
                contracts=contract_items,
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

    def trace_call_chain(
        self, ctx: TraceContext, cmd: TraceCallChainCommand
    ) -> Result[RemoteCallChainResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.trace_call_chain")
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
                err = tree_res.error()
                self._record_span_error(span, err)
                return Result.err(err)

            state = transition_to(state, STATE_FILTERING_NODES)
            tree_data = tree_res.unwrap()
            raw_tree = tree_data.get("tree", [])
            provider_name = str(tree_data.get("provider", "github"))

            from src.features.git_remote_inspector.domain.rules.git_remote_inspector_rules import (
                match_entrypoint_file,
                extract_symbols_and_imports,
            )

            entrypoint_file = match_entrypoint_file(raw_tree, cmd.endpoint_or_symbol)
            chain_nodes: List[CallChainNode] = []
            visited_files = set()
            current_target = entrypoint_file
            current_depth = 1

            while current_target and current_depth <= min(cmd.max_depth, 5):
                if current_target in visited_files:
                    break
                visited_files.add(current_target)

                content_res = self._remote_client.fetch_file_content(
                    ctx=ctx,
                    repo=cmd.repo_url,
                    path=current_target,
                    ref=cmd.ref_name,
                    token=cmd.token,
                )
                if content_res.is_err:
                    break

                file_payload = content_res.unwrap()
                code_text = str(file_payload.get("decoded_content", file_payload.get("content", "")))
                invoked, imports, layer = extract_symbols_and_imports(code_text, current_target)

                node = CallChainNode(
                    depth=current_depth,
                    layer=layer,
                    file_path=current_target,
                    symbol_name=cmd.endpoint_or_symbol if current_depth == 1 else Path(current_target).stem,
                    invoked_symbols=invoked,
                )
                chain_nodes.append(node)

                next_target = None
                for imp in imports:
                    clean_imp = imp.replace(".", "/").split("/")[-1].lower()
                    for item in raw_tree:
                        candidate = str(item.get("path", ""))
                        if candidate not in visited_files and clean_imp in candidate.lower():
                            if any(candidate.endswith(ext) for ext in (".py", ".ts", ".go", ".java", ".rs")):
                                next_target = candidate
                                break
                    if next_target:
                        break

                current_target = next_target
                current_depth += 1

            state = transition_to(state, STATE_COMPLETED)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            result = RemoteCallChainResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
                ref_name=cmd.ref_name,
                target_endpoint_or_symbol=cmd.endpoint_or_symbol,
                entrypoint_file=entrypoint_file,
                call_chain=chain_nodes,
                total_depth_reached=len(chain_nodes),
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


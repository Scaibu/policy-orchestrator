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
    IntegrateContextCommand,
    ExtractedEndpointSignature,
    RemoteIntegrateContextResult,
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
    extract_openapi_endpoint_signature,
    synthesize_endpoint_signature_from_chain,
)
from src.features.git_remote_inspector.domain.machines.git_remote_inspector_machine import (
    STATE_IDLE,
    STATE_FETCHING_TREE,
    STATE_FILTERING_NODES,
    STATE_FETCHING_CONTENT,
    STATE_COMPARING_DIFF,
    STATE_FETCHING_BRANCHES,
    STATE_INTEGRATING_CONTEXT,
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

    def integrate_context(
        self, ctx: TraceContext, cmd: IntegrateContextCommand
    ) -> Result[RemoteIntegrateContextResult, Exception]:
        start_time = time.perf_counter()
        span = None
        state = STATE_IDLE
        try:
            if self._tracer:
                span = self._tracer.start_span("git_remote_inspector.integrate_context")
            state = transition_to(state, STATE_INTEGRATING_CONTEXT)

            commit_sha = cmd.ref_name
            if hasattr(self._repository, "find_cached_endpoint"):
                cached_res = self._repository.find_cached_endpoint(
                    ctx=ctx,
                    repo_url=cmd.repo_url,
                    commit_sha=commit_sha,
                    endpoint_or_entity=cmd.endpoint_or_entity,
                )
                if cached_res.is_ok and cached_res.unwrap():
                    raw_cache = cached_res.unwrap()
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    state = transition_to(state, STATE_COMPLETED)
                    res = RemoteIntegrateContextResult(
                        repo_url=raw_cache.get("repo_url", cmd.repo_url),
                        provider=raw_cache.get("provider", "github"),
                        ref_name=raw_cache.get("ref_name", cmd.ref_name),
                        target_endpoint_or_entity=raw_cache.get("target_endpoint_or_entity", cmd.endpoint_or_entity),
                        match_type=raw_cache.get("match_type", "OPENAPI_SPEC"),
                        contract_file=raw_cache.get("contract_file"),
                        signature=ExtractedEndpointSignature(**raw_cache.get("signature", {})),
                        related_dto_files=raw_cache.get("related_dto_files", []),
                        call_chain=[CallChainNode(**n) for n in raw_cache.get("call_chain", [])],
                        cached=True,
                        duration_ms=duration_ms,
                    )
                    if span and StatusCode:
                        span.set_status(StatusCode.OK)
                    return Result.ok(res)

            disc_cmd = DiscoverContractsCommand(
                repo_url=cmd.repo_url,
                ref_name=cmd.ref_name,
                token=cmd.token,
            )
            disc_res = self.discover_contracts(ctx, disc_cmd)
            provider_name = "github"
            openapi_contract = None
            related_dtos: List[str] = []

            if disc_res.is_ok:
                disc_data = disc_res.unwrap()
                provider_name = disc_data.provider
                if disc_data.commit_sha:
                    commit_sha = disc_data.commit_sha
                    if commit_sha != cmd.ref_name and hasattr(self._repository, "find_cached_endpoint"):
                        sha_cached = self._repository.find_cached_endpoint(
                            ctx=ctx,
                            repo_url=cmd.repo_url,
                            commit_sha=commit_sha,
                            endpoint_or_entity=cmd.endpoint_or_entity,
                        )
                        if sha_cached.is_ok and sha_cached.unwrap():
                            raw_cache = sha_cached.unwrap()
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            state = transition_to(state, STATE_COMPLETED)
                            res = RemoteIntegrateContextResult(
                                repo_url=raw_cache.get("repo_url", cmd.repo_url),
                                provider=raw_cache.get("provider", "github"),
                                ref_name=raw_cache.get("ref_name", cmd.ref_name),
                                target_endpoint_or_entity=raw_cache.get("target_endpoint_or_entity", cmd.endpoint_or_entity),
                                match_type=raw_cache.get("match_type", "OPENAPI_SPEC"),
                                contract_file=raw_cache.get("contract_file"),
                                signature=ExtractedEndpointSignature(**raw_cache.get("signature", {})),
                                related_dto_files=raw_cache.get("related_dto_files", []),
                                call_chain=[CallChainNode(**n) for n in raw_cache.get("call_chain", [])],
                                cached=True,
                                duration_ms=duration_ms,
                            )
                            if span and StatusCode:
                                span.set_status(StatusCode.OK)
                            return Result.ok(res)
                for contract in disc_data.contracts:
                    if cmd.subpath and not contract.path.startswith(cmd.subpath.strip("/")):
                        continue
                    if contract.category == "API_SPECIFICATION" and not openapi_contract:
                        openapi_contract = contract.path
                    elif contract.category == "DATA_CONTRACT_TYPES":
                        related_dtos.append(contract.path)

            if openapi_contract:
                content_cmd = GetFileContentCommand(
                    repo_url=cmd.repo_url,
                    file_path=openapi_contract,
                    ref_name=cmd.ref_name,
                    token=cmd.token,
                )
                content_res = self.get_file_content(ctx, content_cmd)
                if content_res.is_ok:
                    spec_text = content_res.unwrap().content
                    parsed_sig = extract_openapi_endpoint_signature(spec_text, cmd.endpoint_or_entity)
                    if parsed_sig:
                        sig = ExtractedEndpointSignature(**parsed_sig)
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        state = transition_to(state, STATE_COMPLETED)
                        res = RemoteIntegrateContextResult(
                            repo_url=cmd.repo_url,
                            provider=provider_name,
                            ref_name=cmd.ref_name,
                            target_endpoint_or_entity=cmd.endpoint_or_entity,
                            match_type="OPENAPI_SPEC",
                            contract_file=openapi_contract,
                            signature=sig,
                            related_dto_files=related_dtos[:10],
                            call_chain=[],
                            cached=False,
                            duration_ms=duration_ms,
                        )
                        if hasattr(self._repository, "save_endpoint_relationship"):
                            self._repository.save_endpoint_relationship(
                                ctx=ctx,
                                repo_url=cmd.repo_url,
                                commit_sha=commit_sha,
                                ref_name=cmd.ref_name,
                                result_data=res.model_dump(),
                            )
                        if span and StatusCode:
                            span.set_status(StatusCode.OK)
                        return Result.ok(res)

            chain_cmd = TraceCallChainCommand(
                repo_url=cmd.repo_url,
                endpoint_or_symbol=cmd.endpoint_or_entity,
                ref_name=cmd.ref_name,
                max_depth=3,
                token=cmd.token,
            )
            chain_res = self.trace_call_chain(ctx, chain_cmd)
            chain_nodes: List[CallChainNode] = []
            code_blobs: Dict[str, str] = {}

            if chain_res.is_ok:
                chain_data = chain_res.unwrap()
                provider_name = chain_data.provider
                chain_nodes = chain_data.call_chain
                for node in chain_nodes[:3]:
                    c_res = self.get_file_content(
                        ctx,
                        GetFileContentCommand(
                            repo_url=cmd.repo_url,
                            file_path=node.file_path,
                            ref_name=cmd.ref_name,
                            token=cmd.token,
                        ),
                    )
                    if c_res.is_ok:
                        code_blobs[node.file_path] = c_res.unwrap().content

            synthesized = synthesize_endpoint_signature_from_chain(
                target=cmd.endpoint_or_entity,
                chain_nodes=[n.model_dump() for n in chain_nodes],
                code_blobs=code_blobs,
            )
            sig = ExtractedEndpointSignature(**synthesized)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            state = transition_to(state, STATE_COMPLETED)

            res = RemoteIntegrateContextResult(
                repo_url=cmd.repo_url,
                provider=provider_name,
                ref_name=cmd.ref_name,
                target_endpoint_or_entity=cmd.endpoint_or_entity,
                match_type="STATIC_CALL_CHAIN",
                contract_file=chain_nodes[0].file_path if chain_nodes else None,
                signature=sig,
                related_dto_files=related_dtos[:10],
                call_chain=chain_nodes,
                cached=False,
                duration_ms=duration_ms,
            )
            if hasattr(self._repository, "save_endpoint_relationship"):
                self._repository.save_endpoint_relationship(
                    ctx=ctx,
                    repo_url=cmd.repo_url,
                    commit_sha=commit_sha,
                    ref_name=cmd.ref_name,
                    result_data=res.model_dump(),
                )
            if span and StatusCode:
                span.set_status(StatusCode.OK)
            return Result.ok(res)
        except Exception as exc:
            state = STATE_FAILED
            self._record_span_error(span, exc)
            return Result.err(exc)
        finally:
            if span:
                span.end()



from pathlib import Path
from typing import Optional
from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.domain.ports.graph_port import GraphNode, GraphRelationship
from src.features.folder_scaffold.service.folder_scaffold_service_port import FolderScaffoldServicePort
from src.features.folder_scaffold.types.folder_scaffold_deps import FolderScaffoldServiceDeps
from src.features.folder_scaffold.types.folder_scaffold_types import (
    CreateFileCommand,
    LinkFilesCommand,
    CreateFileResult,
    LinkFilesResult,
    ScaffoldTreeCommand,
    ScaffoldTreeResult,
    ScaffoldedFileNode,
    ScaffoldedRelationshipRecord,
)
from src.features.folder_scaffold.rules.folder_scaffold_rules import (
    can_link,
    infer_layer,
    infer_feature_name,
)
from src.features.folder_scaffold.machines.folder_scaffold_machine import (
    STATE_PENDING,
    STATE_VALIDATED,
    STATE_WRITTEN,
    STATE_SYNCED,
    STATE_COMPLETED,
    transition_to,
)
from src.shared.constants.scaffold_constants import (
    DIRECTION_OUTGOING,
    GRAPH_LABEL_FILE_NODE,
    STATUS_SUCCESS,
)

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None


class FolderScaffoldDomainService(AbstractTracedComponent, FolderScaffoldServicePort):
    def __init__(self, deps: FolderScaffoldServiceDeps) -> None:
        super().__init__()
        self._repository = deps.repository
        self._graph_store = deps.graph_store
        self._scaffold_config = deps.scaffold_config


    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def createFile(
        self, ctx: TraceContext, cmd: CreateFileCommand
    ) -> Result[CreateFileResult, Exception]:
        span = None
        if self._tracer:
            span = self._tracer.start_span("folder_scaffold.create_file")
        try:
            state = STATE_PENDING
            if not can_link(cmd.rel_type):
                raise ValueError(f"Invalid relationship type: {cmd.rel_type}")
            state = transition_to(state, STATE_VALIDATED)

            p = Path(cmd.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            created_on_disk = False
            if not p.exists():
                content = cmd.content if cmd.content is not None else ""
                p.write_text(content, encoding="utf-8")
                created_on_disk = True
            size_bytes = p.stat().st_size if p.exists() else 0
            state = transition_to(state, STATE_WRITTEN)

            feature_name = cmd.feature_name or infer_feature_name(cmd.path)
            node_id = str(p.resolve())
            layer = infer_layer(cmd.role)

            node = ScaffoldedFileNode(
                id=node_id,
                file_path=str(p),
                role=cmd.role,
                layer=layer,
                feature_name=feature_name,
                package_name=cmd.package_name,
                size_bytes=size_bytes,
            )

            target_id = str(Path(cmd.target).resolve()) if "/" in cmd.target or "." in cmd.target else cmd.target
            source_id = node_id if cmd.direction == DIRECTION_OUTGOING else target_id
            final_target_id = target_id if cmd.direction == DIRECTION_OUTGOING else node_id

            rel_record = ScaffoldedRelationshipRecord(
                id=f"{source_id}-{cmd.rel_type}-{final_target_id}",
                source_id=source_id,
                target_id=final_target_id,
                rel_type=cmd.rel_type.upper(),
                properties={"direction": cmd.direction, "created_at": "ISO_NOW"},
            )

            self._repository.save_node(node)
            self._repository.save_relationship(rel_record)

            g_node = GraphNode(
                id=node.id,
                label=GRAPH_LABEL_FILE_NODE,
                properties={
                    "file_path": node.file_path,
                    "role": node.role,
                    "layer": node.layer,
                    "feature_name": node.feature_name or "",
                    "package_name": node.package_name or "",
                },
            )
            target_g_node = GraphNode(
                id=target_id,
                label=GRAPH_LABEL_FILE_NODE,
                properties={"file_path": cmd.target},
            )
            g_rel = GraphRelationship(
                source_id=source_id,
                target_id=final_target_id,
                rel_type=cmd.rel_type.upper(),
                properties={"direction": cmd.direction},
            )

            self._graph_store.upsert_node(g_node)
            self._graph_store.upsert_node(target_g_node)
            self._graph_store.upsert_relationship(g_rel)
            state = transition_to(state, STATE_SYNCED)
            state = transition_to(state, STATE_COMPLETED)

            res = CreateFileResult(
                node=node,
                relationship=rel_record,
                created_on_disk=created_on_disk,
                status=state,
            )
            if span and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            return Result.ok(res)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            if span and Status is not None and StatusCode is not None:
                span.record_exception(mapped)
                span.set_status(Status(StatusCode.ERROR, str(mapped)))
            return Result.err(mapped)
        finally:
            if span:
                span.end()

    def linkFiles(
        self, ctx: TraceContext, cmd: LinkFilesCommand
    ) -> Result[LinkFilesResult, Exception]:
        span = None
        if self._tracer:
            span = self._tracer.start_span("folder_scaffold.link_files")
        try:
            if not can_link(cmd.rel_type):
                raise ValueError(f"Invalid relationship type: {cmd.rel_type}")

            source_id = str(Path(cmd.source).resolve()) if "/" in cmd.source or "." in cmd.source else cmd.source
            target_id = str(Path(cmd.target).resolve()) if "/" in cmd.target or "." in cmd.target else cmd.target

            rel_record = ScaffoldedRelationshipRecord(
                id=f"{source_id}-{cmd.rel_type}-{target_id}",
                source_id=source_id,
                target_id=target_id,
                rel_type=cmd.rel_type.upper(),
                properties=cmd.properties or {},
            )

            self._repository.save_relationship(rel_record)

            self._graph_store.upsert_node(GraphNode(id=source_id, label=GRAPH_LABEL_FILE_NODE, properties={"id": source_id}))
            self._graph_store.upsert_node(GraphNode(id=target_id, label=GRAPH_LABEL_FILE_NODE, properties={"id": target_id}))
            self._graph_store.upsert_relationship(
                GraphRelationship(
                    source_id=source_id,
                    target_id=target_id,
                    rel_type=cmd.rel_type.upper(),
                    properties=cmd.properties or {},
                )
            )

            res = LinkFilesResult(
                relationship=rel_record,
                status=STATUS_SUCCESS,
            )
            if span and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            return Result.ok(res)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            if span and Status is not None and StatusCode is not None:
                span.record_exception(mapped)
                span.set_status(Status(StatusCode.ERROR, str(mapped)))
            return Result.err(mapped)
        finally:
            if span:
                span.end()

    def scaffoldTree(
        self, ctx: TraceContext, cmd: ScaffoldTreeCommand
    ) -> Result[ScaffoldTreeResult, Exception]:
        span = None
        if self._tracer:
            span = self._tracer.start_span("folder_scaffold.scaffold_tree")
        try:
            root_path = Path(cmd.root_id)
            root_id = str(root_path.resolve()) if ("/" in cmd.root_id or "." in cmd.root_id) else cmd.root_id

            root_node = ScaffoldedFileNode(
                id=root_id,
                file_path=str(root_path),
                role=cmd.root_role,
                layer="ROOT",
                feature_name=None,
                package_name=cmd.root_id,
                size_bytes=0,
            )
            self._repository.save_node(root_node)
            self._graph_store.upsert_node(
                GraphNode(
                    id=root_id,
                    label="PackageRoot" if cmd.root_role == "PackageRoot" else GRAPH_LABEL_FILE_NODE,
                    properties={"id": root_id, "file_path": str(root_path), "role": cmd.root_role},
                )
            )

            created_nodes: list[ScaffoldedFileNode] = []
            created_relationships: list[ScaffoldedRelationshipRecord] = []

            for node_spec in cmd.nodes:
                target_ref = node_spec.target
                if target_ref in ("$ROOT", "root", cmd.root_id):
                    target_ref = root_id

                file_cmd = CreateFileCommand(
                    path=node_spec.path,
                    role=node_spec.role,
                    rel_type=node_spec.rel_type,
                    target=target_ref,
                    direction=node_spec.direction,
                    feature_name=node_spec.feature_name,
                    package_name=node_spec.package_name or cmd.root_id,
                    content=node_spec.content,
                )
                res = self.createFile(ctx, file_cmd)
                if res.is_err:
                    raise res.error()
                c_res = res.unwrap()
                created_nodes.append(c_res.node)
                created_relationships.append(c_res.relationship)

            for rel_spec in cmd.relationships:
                src_ref = rel_spec.source
                tgt_ref = rel_spec.target
                if src_ref in ("$ROOT", "root", cmd.root_id):
                    src_ref = root_id
                if tgt_ref in ("$ROOT", "root", cmd.root_id):
                    tgt_ref = root_id

                link_cmd = LinkFilesCommand(
                    source=src_ref,
                    rel_type=rel_spec.rel_type,
                    target=tgt_ref,
                    properties=rel_spec.properties or {},
                )
                l_res = self.linkFiles(ctx, link_cmd)
                if l_res.is_err:
                    raise l_res.error()
                created_relationships.append(l_res.unwrap().relationship)

            summary = ScaffoldTreeResult(
                root_id=root_id,
                total_created_files=len(created_nodes),
                total_relationships=len(created_relationships),
                nodes=created_nodes,
                relationships=created_relationships,
                status=STATUS_SUCCESS,
            )
            if span and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            return Result.ok(summary)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            if span and Status is not None and StatusCode is not None:
                span.record_exception(mapped)
                span.set_status(Status(StatusCode.ERROR, str(mapped)))
            return Result.err(mapped)
        finally:
            if span:
                span.end()

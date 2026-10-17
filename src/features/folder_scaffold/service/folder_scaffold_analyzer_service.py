"""
================================================================================
BLUEPRINT: FOLDER SCAFFOLD KNOWLEDGE GRAPH ANALYZER DOMAIN SERVICE
================================================================================
Implements graph analysis queries across AlloyDB and Neo4j:
- Bidirectional dependency traversal (BFS up to max_depth)
- Upstream caller blast radius and risk scoring
- Direct callers and dependents extraction
- Fully decoupled from hardcoded paths or folder layouts
================================================================================
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time

try:
    from opentelemetry.trace import Status, StatusCode
except ImportError:
    Status = None
    StatusCode = None

from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.domain.ports.graph_port import GraphStorePort
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service_port import (
    FolderScaffoldAnalyzerServicePort,
)
from src.features.folder_scaffold.infrastructure.ports.folder_scaffold_repository_port import (
    FolderScaffoldRepositoryPort,
)
from src.features.folder_scaffold.contracts.folder_scaffold_types import (
    BlastRadiusReport,
    FileLineageReport,
    ImpactNodeDetail,
)
from src.features.folder_scaffold.infrastructure.queries.folder_scaffold_graph_queries import (
    build_graph_impact_analysis_query,
    FLOW_GRAPH_LINEAGE_DIRECT_UPSTREAM,
    FLOW_GRAPH_LINEAGE_DIRECT_DOWNSTREAM,
)


class FolderScaffoldAnalyzerDomainService(AbstractTracedComponent, FolderScaffoldAnalyzerServicePort):
    def __init__(
        self,
        repository: FolderScaffoldRepositoryPort,
        graph_store: GraphStorePort,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._graph_store = graph_store

    def mapFailure(self, cause: Exception) -> Exception:
        return cause

    def analyze_impact(
        self,
        ctx: TraceContext,
        target_id: str,
        direction: str = "UPSTREAM",
        max_depth: int = 5,
    ) -> Result[BlastRadiusReport, Exception]:
        span = None
        if self._tracer:
            span = self._tracer.start_span("folder_scaffold.analyze_impact")
        try:
            start = time.perf_counter()
            normalized_id = (
                str(Path(target_id).resolve()) if "/" in target_id or "." in target_id else target_id
            )

            direction_clean = direction.upper().strip()
            cypher = build_graph_impact_analysis_query(direction_clean, max_depth)
            res = self._graph_store.query_cypher(cypher, {"target_id": normalized_id})
            impacted_nodes: List[ImpactNodeDetail] = []
            seen_ids = set()

            for rec in res.records:
                n_id = rec.get("id")
                if not n_id or n_id in seen_ids:
                    continue
                seen_ids.add(n_id)

                depth = int(rec.get("depth", 1))
                risk = "HIGH" if depth <= 2 else "MEDIUM" if depth <= 4 else "LOW"

                impacted_nodes.append(
                    ImpactNodeDetail(
                        id=str(n_id),
                        file_path=str(rec.get("file_path", n_id)),
                        role=str(rec.get("role", "Unknown")),
                        layer=str(rec.get("layer", "Unknown")),
                        depth=depth,
                        relationship=str(rec.get("relationship", "DEPENDS_ON")),
                        risk_factor=risk,
                    )
                )

            total = len(impacted_nodes)
            risk_level = "CRITICAL" if total > 20 else "HIGH" if total > 10 else "MEDIUM" if total > 3 else "LOW"
            blast_score = round(min(1.0, (total * 0.05)), 2)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            report = BlastRadiusReport(
                target_id=normalized_id,
                direction=direction_clean,
                max_depth=max_depth,
                risk_level=risk_level,
                blast_radius_score=blast_score,
                impacted_nodes=impacted_nodes,
                total_impacted=total,
                execution_time_ms=elapsed_ms,
            )
            if span and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            return Result.ok(report)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            if span and Status is not None and StatusCode is not None:
                span.record_exception(mapped)
                span.set_status(Status(StatusCode.ERROR, str(mapped)))
            return Result.err(mapped)
        finally:
            if span:
                span.end()

    def get_file_lineage(
        self,
        ctx: TraceContext,
        file_path: str,
        max_depth: int = 5,
    ) -> Result[FileLineageReport, Exception]:
        span = None
        if self._tracer:
            span = self._tracer.start_span("folder_scaffold.get_file_lineage")
        try:
            normalized_id = (
                str(Path(file_path).resolve()) if "/" in file_path or "." in file_path else file_path
            )

            up_res = self._graph_store.query_cypher(
                FLOW_GRAPH_LINEAGE_DIRECT_UPSTREAM, {"target_id": normalized_id}
            )
            down_res = self._graph_store.query_cypher(
                FLOW_GRAPH_LINEAGE_DIRECT_DOWNSTREAM, {"target_id": normalized_id}
            )

            direct_upstream = [
                {
                    "id": r.get("id"),
                    "file_path": r.get("file_path"),
                    "role": r.get("role"),
                    "layer": r.get("layer"),
                    "rel_type": r.get("rel_type"),
                }
                for r in up_res.records
            ]

            direct_downstream = [
                {
                    "id": r.get("id"),
                    "file_path": r.get("file_path"),
                    "role": r.get("role"),
                    "layer": r.get("layer"),
                    "rel_type": r.get("rel_type"),
                }
                for r in down_res.records
            ]

            stored_node = self._repository.find_node_by_id(normalized_id)
            role = stored_node.role if stored_node else "File"
            layer = stored_node.layer if stored_node else "UNKNOWN"

            impact_res = self.analyze_impact(ctx, normalized_id, direction="UPSTREAM", max_depth=max_depth)
            blast_score = impact_res.unwrap().blast_radius_score if impact_res.is_ok else 0.0

            lineage = FileLineageReport(
                target_id=normalized_id,
                file_path=file_path,
                role=role,
                layer=layer,
                direct_upstream=direct_upstream,
                direct_downstream=direct_downstream,
                blast_radius_score=blast_score,
                is_root_ingress=len(direct_upstream) == 0,
                is_leaf_node=len(direct_downstream) == 0,
            )
            if span and Status is not None and StatusCode is not None:
                span.set_status(Status(StatusCode.OK))
            return Result.ok(lineage)
        except Exception as exc:
            mapped = self.mapFailure(exc)
            if span and Status is not None and StatusCode is not None:
                span.record_exception(mapped)
                span.set_status(Status(StatusCode.ERROR, str(mapped)))
            return Result.err(mapped)
        finally:
            if span:
                span.end()

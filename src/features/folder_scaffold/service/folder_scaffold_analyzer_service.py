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

from src.shared.types.envelope_types import TraceContext
from src.shared.kernel.result import Result
from src.shared.kernel.abstract_traced_component import AbstractTracedComponent
from src.domain.ports.graph_port import GraphStorePort
from src.features.folder_scaffold.service.folder_scaffold_analyzer_service_port import (
    FolderScaffoldAnalyzerServicePort,
)
from src.features.folder_scaffold.repository.folder_scaffold_repository_port import (
    FolderScaffoldRepositoryPort,
)
from src.features.folder_scaffold.types.folder_scaffold_types import (
    BlastRadiusReport,
    FileLineageReport,
    ImpactNodeDetail,
)


class FolderScaffoldAnalyzerDomainService(AbstractTracedComponent, FolderScaffoldAnalyzerServicePort):
    def __init__(
        self,
        repository: FolderScaffoldRepositoryPort,
        graph_store: GraphStorePort,
    ) -> None:
        super().__init__(component_name="FolderScaffoldAnalyzerDomainService")
        self._repository = repository
        self._graph_store = graph_store

    def analyze_impact(
        self,
        ctx: TraceContext,
        target_id: str,
        direction: str = "UPSTREAM",
        max_depth: int = 5,
    ) -> Result[BlastRadiusReport, Exception]:
        start = time.perf_counter()
        normalized_id = (
            str(Path(target_id).resolve()) if "/" in target_id or "." in target_id else target_id
        )

        direction_clean = direction.upper().strip()
        arrow_in = "<-[r*1.." + str(max_depth) + "]-" if direction_clean == "UPSTREAM" else "-[r*1.." + str(max_depth) + "]->"

        cypher = f"""
        MATCH path = (target:FileNode {{id: $target_id}}){arrow_in}(impacted:FileNode)
        RETURN
            impacted.id AS id,
            impacted.file_path AS file_path,
            impacted.role AS role,
            impacted.layer AS layer,
            length(path) AS depth,
            [rel in relationships(path) | type(rel)][0] AS relationship
        ORDER BY depth ASC
        LIMIT 200
        """

        try:
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
            return Result.ok(report)
        except Exception as exc:
            return Result.err(self.mapFailure(exc))

    def get_file_lineage(
        self,
        ctx: TraceContext,
        file_path: str,
        max_depth: int = 5,
    ) -> Result[FileLineageReport, Exception]:
        normalized_id = (
            str(Path(file_path).resolve()) if "/" in file_path or "." in file_path else file_path
        )

        upstream_cypher = """
        MATCH (target:FileNode {id: $target_id})<-[r]-(caller:FileNode)
        RETURN caller.id AS id, caller.file_path AS file_path, caller.role AS role, caller.layer AS layer, type(r) AS rel_type
        LIMIT 50
        """

        downstream_cypher = """
        MATCH (target:FileNode {id: $target_id})-[r]->(dep:FileNode)
        RETURN dep.id AS id, dep.file_path AS file_path, dep.role AS role, dep.layer AS layer, type(r) AS rel_type
        LIMIT 50
        """

        try:
            up_res = self._graph_store.query_cypher(upstream_cypher, {"target_id": normalized_id})
            down_res = self._graph_store.query_cypher(downstream_cypher, {"target_id": normalized_id})

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
            return Result.ok(lineage)
        except Exception as exc:
            return Result.err(self.mapFailure(exc))

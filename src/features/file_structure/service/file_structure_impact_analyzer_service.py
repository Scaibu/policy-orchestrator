from typing import List, Dict, Any, Set, Optional, Tuple
import networkx as nx
from src.domain.ports.graph_port import GraphStorePort
from src.domain.ports.impact_analysis_port import (
    ImpactedNodeModel,
    ImpactAnalysisReportModel,
)
from src.features.file_structure.types.file_structure_types import (
    DEFAULT_MAX_TRAVERSAL_DEPTH,
    WEIGHT_CONTRACT_NODE,
    WEIGHT_HANDLER_NODE,
    WEIGHT_DB_MIGRATION_NODE,
    WEIGHT_SERVICE_NODE,
    WEIGHT_PORT_NODE,
    WEIGHT_DEFAULT_NODE,
    RISK_SCORE_THRESHOLD_HIGH,
    RISK_SCORE_THRESHOLD_MEDIUM,
    RISK_LEVEL_LEVEL_0,
    RISK_LEVEL_LEVEL_1,
    RISK_LEVEL_LEVEL_2,
    RISK_LEVEL_LEVEL_3,
    CRITICAL_STORAGE_LABELS,
    ISOLATED_LOGIC_LABELS,
)

NODE_WEIGHT_REGISTRY: Dict[str, float] = {
    "Contract": WEIGHT_CONTRACT_NODE,
    "IngressHandler": WEIGHT_HANDLER_NODE,
    "IngressRouter": WEIGHT_HANDLER_NODE,
    "DatabaseMigration": WEIGHT_DB_MIGRATION_NODE,
    "NamedQuery": WEIGHT_DB_MIGRATION_NODE,
    "DomainService": WEIGHT_SERVICE_NODE,
    "RepositoryPort": WEIGHT_PORT_NODE,
    "SharedDomainPort": WEIGHT_PORT_NODE,
}

class FileStructureImpactAnalyzerService:
    def __init__(self, graph_store: GraphStorePort) -> None:
        self.graph_store = graph_store

    def analyze_upstream_impact(
        self,
        target_id: str,
        max_depth: int = DEFAULT_MAX_TRAVERSAL_DEPTH,
    ) -> ImpactAnalysisReportModel:
        return self._execute_impact_analysis(
            target_id=target_id,
            direction="UPSTREAM",
            max_depth=max_depth,
        )

    def analyze_downstream_impact(
        self,
        target_id: str,
        max_depth: int = DEFAULT_MAX_TRAVERSAL_DEPTH,
    ) -> ImpactAnalysisReportModel:
        return self._execute_impact_analysis(
            target_id=target_id,
            direction="DOWNSTREAM",
            max_depth=max_depth,
        )

    def _execute_impact_analysis(
        self,
        target_id: str,
        direction: str,
        max_depth: int,
    ) -> ImpactAnalysisReportModel:
        if not target_id:
            raise ValueError("target_id must not be empty")

        resolved_id = self._resolve_target_node_id(target_id)
        graph = self._build_directed_graph()
        traversal_graph = graph.reverse(copy=False) if direction == "UPSTREAM" else graph

        target_label = self._get_node_label(resolved_id, graph)
        affected_layers: Set[str] = {target_label}

        impacted_nodes, total_blast_score = self._compute_traversal_impact(
            traversal_graph=traversal_graph,
            resolved_id=resolved_id,
            max_depth=max_depth,
            affected_layers=affected_layers,
        )

        risk_level = self._resolve_risk_level(target_label, total_blast_score)
        breaking_risks = self._compute_potential_breaking_risks(target_label, impacted_nodes)
        verification_cmds = self._generate_verification_commands(target_label, impacted_nodes)
        mitigation_recipe = self._generate_mitigation_recipe(target_label, impacted_nodes)

        return ImpactAnalysisReportModel(
            target_id=target_id,
            target_label=target_label,
            direction=direction,
            blast_radius_score=round(total_blast_score, 3),
            risk_level=risk_level,
            impacted_nodes=impacted_nodes,
            affected_layers=sorted(list(affected_layers)),
            potential_breaking_risks=breaking_risks,
            required_verification_commands=verification_cmds,
            recommended_mitigation_recipe=mitigation_recipe,
        )

    def _build_directed_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        if hasattr(self.graph_store, "_nodes") and hasattr(self.graph_store, "_relationships"):
            nodes_dict = getattr(self.graph_store, "_nodes")
            relationships_list = getattr(self.graph_store, "_relationships")
            for node_id, node in nodes_dict.items():
                props = getattr(node, "properties", {}) or {}
                graph.add_node(
                    node_id,
                    label=str(getattr(node, "label", "Unknown")),
                    path=str(props.get("file_path", "")),
                )
            for rel in relationships_list:
                graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    rel_type=getattr(rel, "rel_type", "DEPENDS_ON"),
                )
            return graph

        query_res = self.graph_store.query_cypher("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m")
        for node in query_res.nodes:
            props = getattr(node, "properties", {}) or {}
            graph.add_node(
                node.id,
                label=str(getattr(node, "label", "Unknown")),
                path=str(props.get("file_path", "")),
            )
        for rel in query_res.relationships:
            graph.add_edge(
                rel.source_id,
                rel.target_id,
                rel_type=getattr(rel, "rel_type", "DEPENDS_ON"),
            )
        return graph

    def _compute_traversal_impact(
        self,
        traversal_graph: nx.DiGraph,
        resolved_id: str,
        max_depth: int,
        affected_layers: Set[str],
    ) -> Tuple[List[ImpactedNodeModel], float]:
        if resolved_id not in traversal_graph:
            return [], 0.0

        distances = nx.single_source_shortest_path_length(
            traversal_graph,
            source=resolved_id,
            cutoff=max_depth,
        )
        bfs_edges = list(nx.bfs_edges(traversal_graph, source=resolved_id, depth_limit=max_depth))

        visited_nodes: Set[str] = set()
        impacted_nodes: List[ImpactedNodeModel] = []
        total_blast_score: float = 0.0

        for parent_id, current_id in bfs_edges:
            if current_id in visited_nodes:
                continue
            visited_nodes.add(current_id)

            depth = distances.get(current_id, 1)
            node_data = traversal_graph.nodes.get(current_id, {})
            node_label = node_data.get("label") or self._get_node_label(current_id, traversal_graph)
            node_path = node_data.get("path") or self._get_node_path(current_id, traversal_graph)
            edge_data = traversal_graph.get_edge_data(parent_id, current_id) or {}
            rel_type = edge_data.get("rel_type", "DEPENDS_ON")

            affected_layers.add(node_label)
            weight = self._get_node_weight(node_label)
            node_risk = weight / float(depth)
            total_blast_score += node_risk

            impacted_nodes.append(
                ImpactedNodeModel(
                    id=current_id,
                    label=node_label,
                    path=node_path,
                    depth=depth,
                    relationship=rel_type,
                    risk_factor=round(node_risk, 3),
                )
            )

        return impacted_nodes, total_blast_score

    def _resolve_target_node_id(self, target_id: str) -> str:
        node = self.graph_store.find_node(target_id)
        if node:
            return node.id
        return target_id

    def _get_node_label(self, node_id: str, graph: Optional[nx.DiGraph] = None) -> str:
        if graph is not None and node_id in graph.nodes:
            lbl = graph.nodes[node_id].get("label")
            if lbl:
                return str(lbl)
        node = self.graph_store.find_node(node_id)
        if node and node.label:
            return str(node.label)
        return "Unknown"

    def _get_node_path(self, node_id: str, graph: Optional[nx.DiGraph] = None) -> str:
        if graph is not None and node_id in graph.nodes:
            pth = graph.nodes[node_id].get("path")
            if pth:
                return str(pth)
        node = self.graph_store.find_node(node_id)
        if node and node.properties and "file_path" in node.properties:
            return str(node.properties["file_path"])
        return ""

    def _get_node_weight(self, label: str) -> float:
        return NODE_WEIGHT_REGISTRY.get(label, WEIGHT_DEFAULT_NODE)

    def _resolve_risk_level(self, target_label: str, blast_score: float) -> str:
        if target_label in ISOLATED_LOGIC_LABELS:
            return RISK_LEVEL_LEVEL_0
        if target_label == "Contract" or blast_score >= RISK_SCORE_THRESHOLD_HIGH:
            return RISK_LEVEL_LEVEL_3
        if target_label in CRITICAL_STORAGE_LABELS or blast_score >= RISK_SCORE_THRESHOLD_MEDIUM:
            return RISK_LEVEL_LEVEL_2
        return RISK_LEVEL_LEVEL_1

    def _compute_potential_breaking_risks(self, target_label: str, impacted_nodes: List[ImpactedNodeModel]) -> List[str]:
        risks = []
        if target_label in CRITICAL_STORAGE_LABELS:
            risks.append("CRITICAL: Schema drift or column alterations will invalidate upstream repository adapters and SQL queries.")
        if target_label == "Contract":
            risks.append("CRITICAL: Public contract modification poses immediate breaking changes to external REST clients.")
        if any(n.label in ("IngressHandler", "IngressRouter") for n in impacted_nodes):
            risks.append("WARNING: Ingress HTTP handlers and routers depend on this node; status codes or payload envelopes may break.")
        if not risks:
            risks.append("LOW: Changes are isolated within the internal domain feature boundary.")
        return risks

    def _generate_verification_commands(self, target_label: str, impacted_nodes: List[ImpactedNodeModel]) -> List[str]:
        cmds = ["pytest tests/"]
        for node in impacted_nodes:
            if node.label == "Contract":
                cmds.append("npm run test:contracts")
            elif node.label in ("IngressHandler", "IngressRouter"):
                cmds.append(f"pytest tests/api/ -k {node.id}")
            elif node.label == "DatabaseMigration":
                cmds.append("scripts/migrate.sh --dry-run")
        return list(dict.fromkeys(cmds))

    def _generate_mitigation_recipe(self, target_label: str, impacted_nodes: List[ImpactedNodeModel]) -> str:
        if target_label in CRITICAL_STORAGE_LABELS:
            return (
                "MITIGATION RECIPE: 1) Generate forward and rollback DDL in `database/migrations/`. "
                "2) Update named query constants in `queries/{feature}.queries.sql`. "
                "3) Update Repository Adapter hydrating entity schema. "
                "4) Execute migration verification before promoting to production."
            )
        if target_label in ISOLATED_LOGIC_LABELS:
            return (
                "MITIGATION RECIPE: 1) Modify rule or state machine data in `rules/{feature}.rules.py`. "
                "2) Run feature unit tests (`pytest tests/unit/`) to confirm decision matrix correctness."
            )
        return (
            "MITIGATION RECIPE: 1) Apply pure logic change in target file. "
            "2) Execute targeted unit test suite to verify no internal invariants were violated."
        )

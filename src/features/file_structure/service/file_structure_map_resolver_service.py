from collections import deque
from pathlib import Path
from typing import Dict, Any, Optional, List
from src.domain.ports.graph_port import GraphStorePort, GraphNode
from src.features.file_structure.repository.file_structure_query_loader import FileStructureQueryLoader

class FileStructureMapResolverService:
    def __init__(self, graph_store: GraphStorePort) -> None:
        self.graph_store = graph_store

    def resolve_feature_map(self, feature_name: str) -> Dict[str, Any]:
        f = feature_name.lower().replace("-", "_")
        service_node_id = f"service_{f}"
        router_node_id = f"router_{f}"
        handler_node_id = f"handler_{f}"
        route_rules_node_id = f"route_rules_{f}"
        contract_node_id = f"contract_openapi_{f}"
        rules_node_id = f"rules_{f}"
        machine_node_id = f"machine_{f}"
        workflow_node_id = f"workflow_{f}"
        port_node_id = f"port_{f}_repo"
        adapter_node_id = f"adapter_{f}_repo"
        query_node_id = f"query_{f}_sql"
        context_node_id = f"context_{f}"
        index_node_id = f"index_{f}"
        types_node_id = f"types_{f}"
        schema_node_id = f"acl_{f}_schema"
        migration_node_id = f"migration_{f}_0001"
        topic_node_id = f"topic_prod_{f}_events_v1"

        return {
            "feature": f,
            "context": {
                "id": context_node_id,
                "file_path": self._get_node_path(context_node_id) or f"src/features/{f}/context.yml",
            },
            "facade": {
                "id": index_node_id,
                "file_path": self._get_node_path(index_node_id) or f"src/features/{f}/index.py",
            },
            "types": {
                "id": types_node_id,
                "file_path": self._get_node_path(types_node_id) or f"src/features/{f}/types/{f}_types.py",
            },
            "schema": {
                "id": schema_node_id,
                "file_path": self._get_node_path(schema_node_id) or f"src/features/{f}/schema/{f}_schema.py",
            },
            "router": {
                "id": router_node_id,
                "file_path": self._get_node_path(router_node_id) or f"src/api/rest/v1/routers/{f}_router.py",
                "route_prefix": f"/api/v1/{f.replace('_', '-')}",
            },
            "handler": {
                "id": handler_node_id,
                "file_path": self._get_node_path(handler_node_id) or f"src/api/rest/v1/handlers/{f}_handler.py",
            },
            "route_rules": {
                "id": route_rules_node_id,
                "file_path": self._get_node_path(route_rules_node_id) or f"src/api/rest/v1/route.rules/{f}_route_rules.py",
            },
            "service": {
                "id": service_node_id,
                "file_path": self._get_node_path(service_node_id) or f"src/features/{f}/service/{f}_service.py",
            },
            "contract": {
                "id": contract_node_id,
                "file_path": self._get_node_path(contract_node_id) or f"contracts/openapi/{f}.v1.yaml",
            },
            "rules": {
                "id": rules_node_id,
                "file_path": self._get_node_path(rules_node_id) or f"src/features/{f}/rules/{f}_rules.py",
            },
            "machine": {
                "id": machine_node_id,
                "file_path": self._get_node_path(machine_node_id) or f"src/features/{f}/machines/{f}_machine.py",
            },
            "workflow": {
                "id": workflow_node_id,
                "file_path": self._get_node_path(workflow_node_id) or f"src/features/{f}/workflows/{f}_workflow.py",
            },
            "repository": {
                "port": self._get_node_path(port_node_id) or f"src/features/{f}/repository/{f}_repository_port.py",
                "adapter": self._get_node_path(adapter_node_id) or f"src/features/{f}/repository/{f}_repository.py",
                "queries": self._get_node_path(query_node_id) or f"src/features/{f}/queries/{f}.queries.sql",
            },
            "external_connections": {
                "migration": self._get_node_path(migration_node_id) or f"database/migrations/0001_create_{f}_table.sql",
                "topic": self._get_node_path(topic_node_id) or f"messaging/topics/0001_create_{f}_events.json",
            },
        }

    def resolve_package_summary(self, package_name: str) -> Dict[str, Any]:
        pkg_slug = package_name.lower().replace("-", "_")
        pkg_node_id = f"pkg_{pkg_slug}"

        features_list = []
        cypher = FileStructureQueryLoader.get_cypher("FLOW_GET_PACKAGE_FEATURES")
        res = self.graph_store.query_cypher(cypher, {"id": pkg_node_id})

        if res.records:
            for r in res.records:
                features_list.append({
                    "feature_name": r.get("name"),
                    "owner": r.get("owner", f"@{r.get('name')}-team"),
                    "status": r.get("status", "active"),
                    "flags": r.get("flags", f"{r.get('name')}.enabled"),
                    "migrations": r.get("migrations", f"0001_create_{r.get('name')}_table"),
                    "contract_version": r.get("contract_version", "v1"),
                })
        else:
            neighbors = self.graph_store.find_neighbors(pkg_node_id, rel_type="CONTAINS_FEATURE", direction="OUTGOING")
            for node in neighbors:
                p = node.properties or {}
                f_name = p.get("feature_name") or node.id.replace("service_", "")
                features_list.append({
                    "feature_name": f_name,
                    "owner": p.get("owner", f"@{f_name}-team"),
                    "status": p.get("status", "active"),
                    "flags": p.get("flags", f"{f_name}.enabled"),
                    "migrations": p.get("migrations", f"0001_create_{f_name}_table"),
                    "contract_version": p.get("contract_version", "v1"),
                })

        return {
            "package": pkg_slug,
            "package_node_id": pkg_node_id,
            "total_features": len(features_list),
            "features": features_list,
        }

    def resolve_feature_files(self, feature_name: str, package_name: Optional[str] = None) -> Dict[str, Any]:
        f = feature_name.lower().replace("-", "_")
        f_map = self.resolve_feature_map(f)

        files_list = []
        layers_dict: Dict[str, list] = {
            "DomainCore": [],
            "IngressDelivery": [],
            "Behavioral": [],
            "Persistence": [],
            "Contracts": [],
            "Messaging": [],
        }

        canonical_roles = [
            ("ContextBoundary", "DomainCore", f_map["context"]["file_path"]),
            ("PublicFacade", "DomainCore", f_map["facade"]["file_path"]),
            ("DomainTypes", "DomainCore", f_map["types"]["file_path"]),
            ("SchemaACL", "DomainCore", f_map["schema"]["file_path"]),
            ("DomainService", "DomainCore", f_map["service"]["file_path"]),
            ("RuleSet", "Behavioral", f_map["rules"]["file_path"]),
            ("StateMachine", "Behavioral", f_map["machine"]["file_path"]),
            ("WorkflowEngine", "Behavioral", f_map["workflow"]["file_path"]),
            ("RepositoryPort", "Persistence", f_map["repository"]["port"]),
            ("RepositoryAdapter", "Persistence", f_map["repository"]["adapter"]),
            ("NamedQuery", "Persistence", f_map["repository"]["queries"]),
            ("IngressRouter", "IngressDelivery", f_map["router"]["file_path"]),
            ("IngressHandler", "IngressDelivery", f_map["handler"]["file_path"]),
            ("RouteRules", "IngressDelivery", f_map["route_rules"]["file_path"]),
        ]

        for role, layer, path in canonical_roles:
            if path:
                item = {"role": role, "layer": layer, "path": path}
                files_list.append(item)
                layers_dict[layer].append(path)

        return {
            "feature": f,
            "package": package_name or "",
            "total_files": len(files_list),
            "layers": layers_dict,
            "files": files_list,
        }

    def resolve_file_lineage(self, file_path_or_node_id: str, max_depth: int = 5) -> Dict[str, Any]:
        target_node = self._find_node(file_path_or_node_id)
        if not target_node:
            raise ValueError(f"Could not find node or file matching: {file_path_or_node_id}")

        t_id = target_node.id
        t_props = target_node.properties or {}
        t_label = target_node.label or t_props.get("label", "Unknown")
        t_path = t_props.get("file_path", "")
        t_layer = t_props.get("layer", "Unknown")
        t_feature = t_props.get("feature_name", "")
        t_package = t_props.get("package_name", "")

        direct_upstream = []
        incoming_edges = self.graph_store.get_incoming_relationships(t_id)
        for edge in incoming_edges:
            src_node = self._find_node(edge.source_id)
            if src_node:
                sp = src_node.properties or {}
                direct_upstream.append({
                    "node_id": src_node.id,
                    "label": src_node.label,
                    "role": src_node.label,
                    "layer": sp.get("layer", "Unknown"),
                    "file_path": sp.get("file_path", ""),
                    "relationship": edge.rel_type,
                    "direction": "incoming",
                    "description": f"{src_node.label} ({edge.rel_type}) -> {t_label}",
                })

        direct_downstream = []
        outgoing_edges = self.graph_store.get_outgoing_relationships(t_id)
        for edge in outgoing_edges:
            tgt_node = self._find_node(edge.target_id)
            if tgt_node:
                tp = tgt_node.properties or {}
                direct_downstream.append({
                    "node_id": tgt_node.id,
                    "label": tgt_node.label,
                    "role": tgt_node.label,
                    "layer": tp.get("layer", "Unknown"),
                    "file_path": tp.get("file_path", ""),
                    "relationship": edge.rel_type,
                    "direction": "outgoing",
                    "description": f"{t_label} ({edge.rel_type}) -> {tgt_node.label}",
                })

        transitive_upstream = []
        visited_up = {t_id}
        queue_up: deque = deque([(t_id, 0, [])])
        while queue_up:
            curr_id, depth, path = queue_up.popleft()
            if depth > 0:
                node_obj = self._find_node(curr_id)
                if node_obj:
                    np = node_obj.properties or {}
                    transitive_upstream.append({
                        "node_id": node_obj.id,
                        "label": node_obj.label,
                        "layer": np.get("layer", "Unknown"),
                        "file_path": np.get("file_path", ""),
                        "depth": depth,
                        "call_path": path,
                    })
            if depth < max_depth:
                for edge in self.graph_store.get_incoming_relationships(curr_id):
                    if edge.source_id not in visited_up:
                        visited_up.add(edge.source_id)
                        queue_up.append((edge.source_id, depth + 1, path + [edge.rel_type]))

        transitive_downstream = []
        visited_down = {t_id}
        queue_down: deque = deque([(t_id, 0, [])])
        while queue_down:
            curr_id, depth, path = queue_down.popleft()
            if depth > 0:
                node_obj = self._find_node(curr_id)
                if node_obj:
                    np = node_obj.properties or {}
                    transitive_downstream.append({
                        "node_id": node_obj.id,
                        "label": node_obj.label,
                        "layer": np.get("layer", "Unknown"),
                        "file_path": np.get("file_path", ""),
                        "depth": depth,
                        "call_path": path,
                    })
            if depth < max_depth:
                for edge in self.graph_store.get_outgoing_relationships(curr_id):
                    if edge.target_id not in visited_down:
                        visited_down.add(edge.target_id)
                        queue_down.append((edge.target_id, depth + 1, path + [edge.rel_type]))

        return {
            "target": {
                "node_id": t_id,
                "label": t_label,
                "layer": t_layer,
                "file_path": t_path,
                "feature_name": t_feature,
                "package_name": t_package,
                "properties": t_props,
            },
            "summary": {
                "direct_upstream_count": len(direct_upstream),
                "direct_downstream_count": len(direct_downstream),
                "transitive_upstream_count": len(transitive_upstream),
                "transitive_downstream_count": len(transitive_downstream),
            },
            "direct_upstream": direct_upstream,
            "direct_downstream": direct_downstream,
            "transitive_upstream": transitive_upstream,
            "transitive_downstream": transitive_downstream,
        }

    def _find_node(self, file_path_or_node_id: str) -> Optional[GraphNode]:
        return self.graph_store.find_node(file_path_or_node_id)

    def _get_node_path(self, node_id: str) -> str:
        node = self.graph_store.find_node(node_id)
        if node and node.properties and "file_path" in node.properties:
            return str(node.properties["file_path"])
        return ""


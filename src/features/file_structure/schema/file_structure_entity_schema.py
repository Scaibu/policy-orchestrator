from typing import Dict, Any

class FileStructureEntitySchema:
    @staticmethod
    def from_api(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "featureName": payload.get("feature_name") or payload.get("featureName", ""),
            "baseDir": payload.get("base_dir") or payload.get("baseDir", "src/features"),
            "withRouter": bool(payload.get("with_router", True)),
            "withHandler": bool(payload.get("with_handler", True)),
            "portType": payload.get("port_type", "local"),
        }

    @staticmethod
    def to_api(domain_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": domain_result.get("status", "success"),
            "feature": domain_result.get("feature", ""),
            "targetDir": domain_result.get("target_dir", ""),
            "filesCreated": domain_result.get("files_created", []),
            "totalFiles": domain_result.get("total_files", 0),
            "routerPath": domain_result.get("router_path", ""),
            "handlerPath": domain_result.get("handler_path", ""),
            "graphSynced": domain_result.get("graph_synced", True),
        }

KnowledgeGraphEntitySchema = FileStructureEntitySchema

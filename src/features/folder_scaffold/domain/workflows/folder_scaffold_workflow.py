from typing import Dict, Any, List


class FolderScaffoldWorkflow:
    @staticmethod
    def get_steps() -> List[Dict[str, str]]:
        return [
            {"id": "step_validate_input", "role": "validate"},
            {"id": "step_create_disk_files", "role": "filesystem"},
            {"id": "step_persist_alloydb", "role": "persistence"},
            {"id": "step_sync_neo4j", "role": "knowledge_graph"},
        ]

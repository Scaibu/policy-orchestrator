"""
================================================================================
BLUEPRINT: FILE SEARCH AUTOMATION WORKFLOW
================================================================================
1. OVERVIEW:
   Multi-step DAG automation flow for file search operations:
   1. validate: validate directory path and traversal parameters
   2. traverse: execute DFS or BFS recursive tree walk with inode caching
   3. filter: apply pattern and extension match filters
   4. record: record search metrics into repository
2. ZERO-INLINE-COMMENT DOCTRINE:
   All DAG step specifications are defined in this top-level header.
================================================================================
"""
from typing import Dict, List


class FileSearchWorkflow:
    @staticmethod
    def get_steps() -> List[Dict[str, str]]:
        return [
            {"id": "step_validate_input", "role": "validate"},
            {"id": "step_traverse_tree", "role": "filesystem"},
            {"id": "step_filter_results", "role": "filter"},
            {"id": "step_persist_metrics", "role": "persistence"},
        ]

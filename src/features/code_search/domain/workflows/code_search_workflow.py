"""
================================================================================
BLUEPRINT: CODE SEARCH AUTOMATION WORKFLOW
================================================================================
1. OVERVIEW:
   Multi-step DAG automation flow for code search:
   1. validate: validate input patterns and target files list
   2. compile: construct Aho-Corasick automaton or Boyer-Moore shift table
   3. scan: stream file contents and apply pattern matcher
   4. persist: record telemetry to database
================================================================================
"""
from typing import Dict, List


class CodeSearchWorkflow:
    @staticmethod
    def get_steps() -> List[Dict[str, str]]:
        return [
            {"id": "step_validate_input", "role": "validate"},
            {"id": "step_compile_matcher", "role": "compiler"},
            {"id": "step_scan_content", "role": "scanner"},
            {"id": "step_persist_metrics", "role": "persistence"},
        ]

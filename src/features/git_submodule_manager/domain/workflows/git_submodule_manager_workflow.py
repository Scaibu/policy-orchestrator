"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER MULTI-STEP WORKFLOW
================================================================================
1. OVERVIEW & OBJECTIVE:
   Encapsulates multi-step automation DAG steps for submodule operations:
     - Step 1: Verify Parent OCC and HEAD SHA.
     - Step 2: Fetch and Parse .gitmodules from Parent Tree.
     - Step 3: Resolve Child Submodule Target Commit SHA.
     - Step 4: Verify Child Commit on Submodule Remote (Multi-Repo Push Guard).
     - Step 5: Synthesize Updated Merkle Tree with Mode 160000 Gitlinks.
     - Step 6: Create Commit, Push Ref, and Sync Graph.

2. INVARIANTS:
   - ZERO comments inside class bodies or function bodies.
================================================================================
"""

from typing import Dict, Any, List, Optional
from src.features.git_submodule_manager.domain.rules.git_submodule_manager_rules import (
    parse_gitmodules_content,
    upsert_submodule_in_gitmodules,
    build_gitlink_tree_nodes,
    sanitize_submodule_path,
)


class SubmoduleSyncWorkflowDAG:
    def __init__(self, parent_repo: str, branch: str) -> None:
        self.parent_repo = parent_repo
        self.branch = branch

    def prepare_add_nodes(
        self,
        existing_gitmodules: str,
        sub_path: str,
        sub_url: str,
        sub_branch: str,
        child_commit_sha: str,
        gitmodules_blob_sha: str,
    ) -> List[Dict[str, Any]]:
        clean_path = sanitize_submodule_path(sub_path)
        shas = {clean_path: child_commit_sha}
        return build_gitlink_tree_nodes(shas, gitmodules_blob_sha)

    def prepare_update_nodes(
        self,
        submodule_updates: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        return build_gitlink_tree_nodes(submodule_updates)

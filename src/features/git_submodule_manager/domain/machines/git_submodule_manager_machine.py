"""
================================================================================
BLUEPRINT: GIT SUBMODULE MANAGER FINITE STATE MACHINE
================================================================================
1. OVERVIEW & OBJECTIVE:
   Deterministic state machine tracking submodule workflow execution:
     IDLE -> VALIDATING_INPUT -> INSPECTING_CHILD_REMOTES -> PREPARING_TREE_MUTATION -> CREATING_COMMIT -> UPDATING_REF -> SYNCING_GRAPH -> COMPLETED / FAILED

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Strictly validated transition DAG.
================================================================================
"""

from typing import Dict, Set

STATE_IDLE = "IDLE"
STATE_VALIDATING_INPUT = "VALIDATING_INPUT"
STATE_INSPECTING_CHILD_REMOTES = "INSPECTING_CHILD_REMOTES"
STATE_PREPARING_TREE_MUTATION = "PREPARING_TREE_MUTATION"
STATE_CREATING_COMMIT = "CREATING_COMMIT"
STATE_UPDATING_REF = "UPDATING_REF"
STATE_SYNCING_GRAPH = "SYNCING_GRAPH"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

_VALID_TRANSITIONS: Dict[str, Set[str]] = {
    STATE_IDLE: {STATE_VALIDATING_INPUT, STATE_FAILED},
    STATE_VALIDATING_INPUT: {STATE_INSPECTING_CHILD_REMOTES, STATE_PREPARING_TREE_MUTATION, STATE_FAILED},
    STATE_INSPECTING_CHILD_REMOTES: {STATE_PREPARING_TREE_MUTATION, STATE_COMPLETED, STATE_FAILED},
    STATE_PREPARING_TREE_MUTATION: {STATE_CREATING_COMMIT, STATE_COMPLETED, STATE_FAILED},
    STATE_CREATING_COMMIT: {STATE_UPDATING_REF, STATE_FAILED},
    STATE_UPDATING_REF: {STATE_SYNCING_GRAPH, STATE_FAILED},
    STATE_SYNCING_GRAPH: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current_state: str, next_state: str) -> str:
    valid_targets = _VALID_TRANSITIONS.get(current_state, set())
    if next_state not in valid_targets:
        raise ValueError(f"Invalid state transition from {current_state} to {next_state}")
    return next_state

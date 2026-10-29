"""
================================================================================
BLUEPRINT: GIT BRANCH LIFECYCLE FINITE STATE MACHINE
================================================================================
1. OVERVIEW & OBJECTIVE:
   Deterministic state machine tracking branch lifecycle operation stages:
     IDLE -> VALIDATING_INPUT -> EXECUTING_PROVIDER -> SYNCING_GRAPH -> AUDITING -> COMPLETED / FAILED

2. INVARIANTS:
   - ZERO comments inside function bodies.
   - Strictly validated state transitions.
================================================================================
"""

from typing import Dict, Set

STATE_IDLE = "IDLE"
STATE_VALIDATING_INPUT = "VALIDATING_INPUT"
STATE_EXECUTING_PROVIDER = "EXECUTING_PROVIDER"
STATE_SYNCING_GRAPH = "SYNCING_GRAPH"
STATE_AUDITING = "AUDITING"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

_VALID_TRANSITIONS: Dict[str, Set[str]] = {
    STATE_IDLE: {STATE_VALIDATING_INPUT, STATE_FAILED},
    STATE_VALIDATING_INPUT: {STATE_EXECUTING_PROVIDER, STATE_FAILED},
    STATE_EXECUTING_PROVIDER: {STATE_SYNCING_GRAPH, STATE_AUDITING, STATE_FAILED},
    STATE_SYNCING_GRAPH: {STATE_AUDITING, STATE_FAILED},
    STATE_AUDITING: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current_state: str, next_state: str) -> str:
    valid_targets = _VALID_TRANSITIONS.get(current_state, set())
    if next_state not in valid_targets:
        raise ValueError(f"Invalid state transition from {current_state} to {next_state}")
    return next_state

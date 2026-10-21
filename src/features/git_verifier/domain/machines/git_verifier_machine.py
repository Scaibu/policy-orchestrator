"""
================================================================================
BLUEPRINT: GIT VERIFIER LIFECYCLE STATE MACHINE
================================================================================
1. OVERVIEW:
   Deterministic state transitions for Git repository & submodule verification:
   PENDING -> READING_HEAD -> INSPECTING_SUBMODULES -> COMPLETED / FAILED.
================================================================================
"""
from typing import Dict, Set


STATE_PENDING = "PENDING"
STATE_READING_HEAD = "READING_HEAD"
STATE_INSPECTING = "INSPECTING_SUBMODULES"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

TRANSITIONS: Dict[str, Set[str]] = {
    STATE_PENDING: {STATE_READING_HEAD, STATE_FAILED},
    STATE_READING_HEAD: {STATE_INSPECTING, STATE_FAILED},
    STATE_INSPECTING: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current: str, target: str) -> str:
    valid_targets = TRANSITIONS.get(current, set())
    if target in valid_targets:
        return target
    return STATE_FAILED

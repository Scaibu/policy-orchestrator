"""
================================================================================
BLUEPRINT: MODULE RESOLVER LIFECYCLE STATE MACHINE
================================================================================
1. OVERVIEW:
   Deterministic state transitions for module resolution:
   PENDING -> RESOLVING_PATH -> CHECKING_COLLISIONS -> COMPLETED / FAILED.
================================================================================
"""
from typing import Dict, Set


STATE_PENDING = "PENDING"
STATE_RESOLVING = "RESOLVING_PATH"
STATE_CHECKING = "CHECKING_COLLISIONS"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

TRANSITIONS: Dict[str, Set[str]] = {
    STATE_PENDING: {STATE_RESOLVING, STATE_FAILED},
    STATE_RESOLVING: {STATE_CHECKING, STATE_FAILED},
    STATE_CHECKING: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current: str, target: str) -> str:
    valid_targets = TRANSITIONS.get(current, set())
    if target in valid_targets:
        return target
    return STATE_FAILED

"""
================================================================================
BLUEPRINT: CODE SEARCH LIFECYCLE STATE MACHINE
================================================================================
1. OVERVIEW:
   Deterministic state transitions for code pattern search:
   PENDING -> COMPILING_AUTOMATON -> SCANNING_FILES -> COMPLETED / FAILED.
2. ZERO-INLINE-COMMENT DOCTRINE:
   State set and transition matrix are declared in this blueprint header.
================================================================================
"""
from typing import Dict, Set


STATE_PENDING = "PENDING"
STATE_COMPILING = "COMPILING_AUTOMATON"
STATE_SCANNING = "SCANNING_FILES"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

TRANSITIONS: Dict[str, Set[str]] = {
    STATE_PENDING: {STATE_COMPILING, STATE_FAILED},
    STATE_COMPILING: {STATE_SCANNING, STATE_FAILED},
    STATE_SCANNING: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current: str, target: str) -> str:
    valid_targets = TRANSITIONS.get(current, set())
    if target in valid_targets:
        return target
    return STATE_FAILED

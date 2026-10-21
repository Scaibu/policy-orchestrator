"""
================================================================================
BLUEPRINT: FILE SEARCH LIFECYCLE STATE MACHINE
================================================================================
1. OVERVIEW:
   Deterministic state transitions for the file search execution lifecycle:
   PENDING -> SCANNING -> FILTERED -> COMPLETED / FAILED.
2. ZERO-INLINE-COMMENT DOCTRINE:
   State set and transition matrix are declared in this blueprint header.
================================================================================
"""
from typing import Dict, Set


STATE_PENDING = "PENDING"
STATE_SCANNING = "SCANNING"
STATE_FILTERED = "FILTERED"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

TRANSITIONS: Dict[str, Set[str]] = {
    STATE_PENDING: {STATE_SCANNING, STATE_FAILED},
    STATE_SCANNING: {STATE_FILTERED, STATE_FAILED},
    STATE_FILTERED: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current: str, target: str) -> str:
    valid_targets = TRANSITIONS.get(current, set())
    if target in valid_targets:
        return target
    return STATE_FAILED

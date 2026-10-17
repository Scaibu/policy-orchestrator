from typing import Dict, Set


STATE_PENDING = "PENDING"
STATE_VALIDATED = "VALIDATED"
STATE_WRITTEN = "WRITTEN"
STATE_SYNCED = "SYNCED"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"

TRANSITIONS: Dict[str, Set[str]] = {
    STATE_PENDING: {STATE_VALIDATED, STATE_FAILED},
    STATE_VALIDATED: {STATE_WRITTEN, STATE_FAILED},
    STATE_WRITTEN: {STATE_SYNCED, STATE_FAILED},
    STATE_SYNCED: {STATE_COMPLETED, STATE_FAILED},
    STATE_COMPLETED: set(),
    STATE_FAILED: set(),
}


def transition_to(current: str, target: str) -> str:
    valid_targets = TRANSITIONS.get(current, set())
    if target in valid_targets:
        return target
    return STATE_FAILED

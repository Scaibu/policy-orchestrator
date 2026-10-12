from src.features.folder_scaffold.machines.folder_scaffold_machine import (
    STATE_PENDING,
    STATE_VALIDATED,
    STATE_WRITTEN,
    STATE_SYNCED,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_state_machine_valid_transitions():
    s1 = transition_to(STATE_PENDING, STATE_VALIDATED)
    assert s1 == STATE_VALIDATED

    s2 = transition_to(s1, STATE_WRITTEN)
    assert s2 == STATE_WRITTEN

    s3 = transition_to(s2, STATE_SYNCED)
    assert s3 == STATE_SYNCED

    s4 = transition_to(s3, STATE_COMPLETED)
    assert s4 == STATE_COMPLETED


def test_state_machine_invalid_transitions():
    s_invalid = transition_to(STATE_PENDING, STATE_COMPLETED)
    assert s_invalid == STATE_FAILED

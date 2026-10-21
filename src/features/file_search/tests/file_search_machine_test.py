from src.features.file_search.domain.machines.file_search_machine import (
    STATE_PENDING,
    STATE_SCANNING,
    STATE_FILTERED,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_file_search_machine_valid_flow():
    s1 = transition_to(STATE_PENDING, STATE_SCANNING)
    assert s1 == STATE_SCANNING

    s2 = transition_to(s1, STATE_FILTERED)
    assert s2 == STATE_FILTERED

    s3 = transition_to(s2, STATE_COMPLETED)
    assert s3 == STATE_COMPLETED


def test_file_search_machine_invalid_transition():
    s = transition_to(STATE_PENDING, STATE_COMPLETED)
    assert s == STATE_FAILED

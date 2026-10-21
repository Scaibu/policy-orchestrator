from src.features.code_search.domain.machines.code_search_machine import (
    STATE_PENDING,
    STATE_COMPILING,
    STATE_SCANNING,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_code_search_machine_valid_flow():
    s1 = transition_to(STATE_PENDING, STATE_COMPILING)
    assert s1 == STATE_COMPILING

    s2 = transition_to(s1, STATE_SCANNING)
    assert s2 == STATE_SCANNING

    s3 = transition_to(s2, STATE_COMPLETED)
    assert s3 == STATE_COMPLETED


def test_code_search_machine_invalid_transition():
    s = transition_to(STATE_PENDING, STATE_COMPLETED)
    assert s == STATE_FAILED

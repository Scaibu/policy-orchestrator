from src.features.git_verifier.domain.machines.git_verifier_machine import (
    STATE_PENDING,
    STATE_READING_HEAD,
    STATE_INSPECTING,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_git_verifier_machine_valid_flow():
    s1 = transition_to(STATE_PENDING, STATE_READING_HEAD)
    assert s1 == STATE_READING_HEAD

    s2 = transition_to(s1, STATE_INSPECTING)
    assert s2 == STATE_INSPECTING

    s3 = transition_to(s2, STATE_COMPLETED)
    assert s3 == STATE_COMPLETED


def test_git_verifier_machine_invalid_transition():
    s = transition_to(STATE_PENDING, STATE_COMPLETED)
    assert s == STATE_FAILED

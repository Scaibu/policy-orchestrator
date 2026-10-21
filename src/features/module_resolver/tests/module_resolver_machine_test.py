from src.features.module_resolver.domain.machines.module_resolver_machine import (
    STATE_PENDING,
    STATE_RESOLVING,
    STATE_CHECKING,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_module_resolver_machine_valid_flow():
    s1 = transition_to(STATE_PENDING, STATE_RESOLVING)
    assert s1 == STATE_RESOLVING

    s2 = transition_to(s1, STATE_CHECKING)
    assert s2 == STATE_CHECKING

    s3 = transition_to(s2, STATE_COMPLETED)
    assert s3 == STATE_COMPLETED


def test_module_resolver_machine_invalid_transition():
    s = transition_to(STATE_PENDING, STATE_COMPLETED)
    assert s == STATE_FAILED

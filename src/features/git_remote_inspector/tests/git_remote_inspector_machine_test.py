import pytest
from src.features.git_remote_inspector.domain.machines.git_remote_inspector_machine import (
    STATE_IDLE,
    STATE_FETCHING_TREE,
    STATE_FILTERING_NODES,
    STATE_FETCHING_CONTENT,
    STATE_COMPARING_DIFF,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


def test_git_remote_inspector_machine_tree_flow():
    s1 = transition_to(STATE_IDLE, STATE_FETCHING_TREE)
    assert s1 == STATE_FETCHING_TREE

    s2 = transition_to(s1, STATE_FILTERING_NODES)
    assert s2 == STATE_FILTERING_NODES

    s3 = transition_to(s2, STATE_COMPLETED)
    assert s3 == STATE_COMPLETED


def test_git_remote_inspector_machine_content_flow():
    s1 = transition_to(STATE_IDLE, STATE_FETCHING_CONTENT)
    assert s1 == STATE_FETCHING_CONTENT

    s2 = transition_to(s1, STATE_COMPLETED)
    assert s2 == STATE_COMPLETED


def test_git_remote_inspector_machine_diff_flow():
    s1 = transition_to(STATE_IDLE, STATE_COMPARING_DIFF)
    assert s1 == STATE_COMPARING_DIFF

    s2 = transition_to(s1, STATE_COMPLETED)
    assert s2 == STATE_COMPLETED


def test_git_remote_inspector_machine_invalid_transition():
    with pytest.raises(ValueError):
        transition_to(STATE_IDLE, STATE_FILTERING_NODES)

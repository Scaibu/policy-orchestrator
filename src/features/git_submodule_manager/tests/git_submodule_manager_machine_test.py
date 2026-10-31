import unittest
from src.features.git_submodule_manager.domain.machines.git_submodule_manager_machine import (
    STATE_IDLE,
    STATE_VALIDATING_INPUT,
    STATE_INSPECTING_CHILD_REMOTES,
    STATE_PREPARING_TREE_MUTATION,
    STATE_CREATING_COMMIT,
    STATE_UPDATING_REF,
    STATE_SYNCING_GRAPH,
    STATE_COMPLETED,
    STATE_FAILED,
    transition_to,
)


class TestGitSubmoduleMachine(unittest.TestCase):
    def test_valid_transition_flow(self):
        state = STATE_IDLE
        state = transition_to(state, STATE_VALIDATING_INPUT)
        state = transition_to(state, STATE_INSPECTING_CHILD_REMOTES)
        state = transition_to(state, STATE_PREPARING_TREE_MUTATION)
        state = transition_to(state, STATE_CREATING_COMMIT)
        state = transition_to(state, STATE_UPDATING_REF)
        state = transition_to(state, STATE_SYNCING_GRAPH)
        state = transition_to(state, STATE_COMPLETED)
        self.assertEqual(state, STATE_COMPLETED)

    def test_invalid_transition_fails(self):
        with self.assertRaises(ValueError):
            transition_to(STATE_IDLE, STATE_COMPLETED)


if __name__ == "__main__":
    unittest.main()

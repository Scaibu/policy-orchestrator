from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDeepQNetworksDqn:
    """
    ---
    contract:
      algo_id: ALGO-NN-196
      name: NnAlgoDeepQNetworksDqn
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - dqn
        - q_learning
        - bellman_equation
        - experience_replay
      inputs:
        type: object
        required:
          - current_q_values
          - action_taken
          - reward
          - next_state_target_q_values
        properties:
          current_q_values:
            type: array
            items:
              type: number
            description: Online network action-value predictions Q(s, :) of length |A|.
          action_taken:
            type: integer
            minimum: 0
            description: Discrete action index a taken in state s.
          reward:
            type: number
            description: Scalar environment reward r received after executing action a.
          next_state_target_q_values:
            type: array
            items:
              type: number
            description: Target network action-value predictions Q_target(s', :) of length |A|.
          gamma_discount:
            type: number
            default: 0.99
            minimum: 0.0
            maximum: 1.0
            description: Discount factor gamma in [0, 1].
          is_terminal:
            type: boolean
            default: false
            description: Boolean flag indicating if next state s' is an episode termination state.
          huber_delta:
            type: number
            default: 1.0
            minimum: 0.001
            description: Threshold delta for Huber / smooth L1 loss.
      outputs:
        type: object
        required:
          - td_target
          - td_error
          - huber_loss
          - greedy_action
        properties:
          td_target:
            type: number
            description: 1-step Bellman target y = r + (1 - done) * gamma * max_a' Q_target(s', a').
          td_error:
            type: number
            description: Temporal difference error delta = y - Q(s, a).
          huber_loss:
            type: number
            description: Robust Huber loss between predicted Q(s, a) and target y.
          greedy_action:
            type: integer
            description: Optimal greedy action argmax_a Q(s, a).
      complexity:
        time: O(|A|)
        space: O(1)
      preconditions:
        - len(input.current_q_values) > 0
        - 0 <= input.action_taken < len(input.current_q_values)
        - len(input.next_state_target_q_values) == len(input.current_q_values)
        - 0.0 <= input.gamma_discount <= 1.0
        - input.huber_delta > 0.0
      postconditions:
        - output.huber_loss >= 0.0
        - 0 <= output.greedy_action < len(input.current_q_values)
      certificate: "TD target exactly equals reward if is_terminal is true"
      compatible_adapters:
        - ADAPTER-DQN-TRAINING-STEPPER
        - ADAPTER-BELLMAN-OPTIMIZER
      related_algos:
        - ALGO-NN-197
      references:
        - "https://doi.org/10.1038/nature14236"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_q_values: List[float],
        action_taken: int,
        reward: float,
        next_state_target_q_values: List[float],
        gamma_discount: float = 0.99,
        is_terminal: bool = False,
        huber_delta: float = 1.0,
    ) -> Dict[str, Any]:
        num_actions = len(current_q_values)
        if num_actions == 0:
            raise ValueError("Precondition failed: current_q_values cannot be empty")
        if not (0 <= action_taken < num_actions):
            raise ValueError(f"Precondition failed: action_taken {action_taken} out of range [0, {num_actions - 1}]")
        if len(next_state_target_q_values) != num_actions:
            raise ValueError("Precondition failed: next_state_target_q_values dimension mismatch")
        if not (0.0 <= gamma_discount <= 1.0) or huber_delta <= 0.0:
            raise ValueError("Precondition failed: invalid gamma_discount or huber_delta")

        # 1. Compute 1-step Bellman target: y = r if terminal else r + gamma * max_a' Q_target(s', a')
        if is_terminal:
            td_target = reward
        else:
            max_next_q = max(next_state_target_q_values)
            td_target = reward + gamma_discount * max_next_q

        # 2. Temporal Difference (TD) error: delta = y - Q(s, a)
        pred_q = current_q_values[action_taken]
        td_error = td_target - pred_q

        # 3. Huber loss:
        # L(delta) = 0.5 * delta^2 if |delta| <= d else d * (|delta| - 0.5 * d)
        abs_err = abs(td_error)
        if abs_err <= huber_delta:
            loss = 0.5 * (td_error ** 2)
        else:
            loss = huber_delta * (abs_err - 0.5 * huber_delta)

        # 4. Greedy action selection
        greedy_a = max(range(num_actions), key=lambda a: current_q_values[a])

        return {
            "td_target": td_target,
            "td_error": td_error,
            "huber_loss": loss,
            "greedy_action": greedy_a,
        }

from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRainbowDqnExtensions:
    """
    ---
    contract:
      algo_id: ALGO-NN-197
      name: NnAlgoRainbowDqnExtensions
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - rainbow_dqn
        - double_dqn
        - dueling_networks
        - prioritized_replay
      inputs:
        type: object
        required:
          - state_value_v
          - advantages_a
          - action_taken
          - reward
          - next_online_q_values
          - next_target_q_values
        properties:
          state_value_v:
            type: number
            description: Dueling network scalar state-value stream V(s).
          advantages_a:
            type: array
            items:
              type: number
            description: Dueling network action-advantage stream A(s, :) of length |A|.
          action_taken:
            type: integer
            minimum: 0
            description: Discrete action index a taken in state s.
          reward:
            type: number
            description: Observed scalar transition reward r.
          next_online_q_values:
            type: array
            items:
              type: number
            description: Next-state online network Q_online(s', :) for action selection.
          next_target_q_values:
            type: array
            items:
              type: number
            description: Next-state target network Q_target(s', :) for action evaluation.
          gamma_discount:
            type: number
            default: 0.99
            minimum: 0.0
            maximum: 1.0
            description: Discount factor gamma in [0, 1].
          is_terminal:
            type: boolean
            default: false
            description: Whether transition reached an episode termination state.
          per_priority_alpha:
            type: number
            default: 0.6
            minimum: 0.0
            maximum: 1.0
            description: Prioritized replay exponent alpha governing prioritization degree.
      outputs:
        type: object
        required:
          - synthesized_q_values
          - double_td_target
          - double_td_error
          - per_transition_priority
          - greedy_action
        properties:
          synthesized_q_values:
            type: array
            items:
              type: number
            description: Mean-centered Dueling Q-values Q(s, :) = V(s) + (A(s, :) - mean(A)).
          double_td_target:
            type: number
            description: Decoupled Double DQN target y = r + gamma * Q_target(s', argmax Q_online(s')).
          double_td_error:
            type: number
            description: Double DQN TD error delta = y - Q(s, a).
          per_transition_priority:
            type: number
            description: Prioritized replay sampling priority p = (|delta| + eps)^alpha.
          greedy_action:
            type: integer
            description: Greedy action selected by synthesized Q-values.
      complexity:
        time: O(|A|)
        space: O(|A|)
      preconditions:
        - len(input.advantages_a) > 0
        - 0 <= input.action_taken < len(input.advantages_a)
        - len(input.next_online_q_values) == len(input.advantages_a)
        - len(input.next_target_q_values) == len(input.advantages_a)
        - 0.0 <= input.gamma_discount <= 1.0
        - 0.0 <= input.per_priority_alpha <= 1.0
      postconditions:
        - len(output.synthesized_q_values) == len(input.advantages_a)
        - output.per_transition_priority > 0.0
        - 0 <= output.greedy_action < len(input.advantages_a)
      certificate: "Mean of advantages subtracted from Q-values satisfies mean(Q - V) == 0.0"
      compatible_adapters:
        - ADAPTER-RAINBOW-STEPPER
        - ADAPTER-DUELING-DOUBLE-DQN
      related_algos:
        - ALGO-NN-196
      references:
        - "https://arxiv.org/abs/1710.02298"
    ---
    """

    @classmethod
    def forward(
        cls,
        state_value_v: float,
        advantages_a: List[float],
        action_taken: int,
        reward: float,
        next_online_q_values: List[float],
        next_target_q_values: List[float],
        gamma_discount: float = 0.99,
        is_terminal: bool = False,
        per_priority_alpha: float = 0.6,
    ) -> Dict[str, Any]:
        num_actions = len(advantages_a)
        if num_actions == 0:
            raise ValueError("Precondition failed: advantages_a cannot be empty")
        if not (0 <= action_taken < num_actions):
            raise ValueError(f"Precondition failed: action_taken {action_taken} out of bounds")
        if len(next_online_q_values) != num_actions or len(next_target_q_values) != num_actions:
            raise ValueError("Precondition failed: next state Q-value dimensions mismatch")
        if not (0.0 <= gamma_discount <= 1.0) or not (0.0 <= per_priority_alpha <= 1.0):
            raise ValueError("Precondition failed: gamma or alpha out of range")

        # 1. Dueling Network centering: Q(s, a) = V(s) + (A(s, a) - (1/|A|) * sum A)
        mean_adv = sum(advantages_a) / float(num_actions)
        q_values: List[float] = [state_value_v + (a - mean_adv) for a in advantages_a]

        # 2. Double DQN target evaluation:
        # Action chosen by online network: a* = argmax_a' Q_online(s', a')
        # Target evaluated by target network: Q_target(s', a*)
        if is_terminal:
            double_target = reward
        else:
            best_next_a = max(range(num_actions), key=lambda a: next_online_q_values[a])
            target_eval_q = next_target_q_values[best_next_a]
            double_target = reward + gamma_discount * target_eval_q

        # 3. Double TD error
        pred_q = q_values[action_taken]
        td_error = double_target - pred_q

        # 4. Prioritized Experience Replay priority: p = (|delta| + eps)^alpha
        eps = 1e-6
        per_priority = (abs(td_error) + eps) ** per_priority_alpha

        greedy_a = max(range(num_actions), key=lambda a: q_values[a])

        return {
            "synthesized_q_values": q_values,
            "double_td_target": double_target,
            "double_td_error": td_error,
            "per_transition_priority": per_priority,
            "greedy_action": greedy_a,
        }

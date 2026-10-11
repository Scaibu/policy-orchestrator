from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPolicyGradientsReinforce:
    """
    ---
    contract:
      algo_id: ALGO-NN-198
      name: NnAlgoPolicyGradientsReinforce
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - policy_gradients
        - reinforce
        - baseline_subtraction
        - monte_carlo_returns
      inputs:
        type: object
        required:
          - action_probabilities
          - step_rewards
          - baseline_values
        properties:
          action_probabilities:
            type: array
            items:
              type: number
            description: Selected action probabilities pi(a_t | s_t) along the episode trajectory of length T.
          step_rewards:
            type: array
            items:
              type: number
            description: Environment rewards r_t obtained at each step of length T.
          baseline_values:
            type: array
            items:
              type: number
            description: Value baseline predictions b_t of length T.
          gamma_discount:
            type: number
            default: 0.99
            minimum: 0.0
            maximum: 1.0
            description: Temporal discount factor gamma.
          normalize_advantages:
            type: boolean
            default: true
            description: Whether to whiten advantages to zero-mean unit-variance per batch.
      outputs:
        type: object
        required:
          - discounted_returns
          - advantages
          - policy_gradient_loss
          - trajectory_return
        properties:
          discounted_returns:
            type: array
            items:
              type: number
            description: Monte Carlo discounted cumulative returns G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k.
          advantages:
            type: array
            items:
              type: number
            description: Baseline-subtracted (and optionally normalized) advantages A_t.
          policy_gradient_loss:
            type: number
            description: Negative expected log-likelihood surrogate loss - (1/T) sum log(pi) * A.
          trajectory_return:
            type: number
            description: Undiscounted cumulative episode return sum_t r_t.
      complexity:
        time: O(T)
        space: O(T)
      preconditions:
        - len(input.action_probabilities) > 0
        - len(input.step_rewards) == len(input.action_probabilities)
        - len(input.baseline_values) == len(input.action_probabilities)
        - all(p > 0.0 for p in input.action_probabilities)
        - 0.0 <= input.gamma_discount <= 1.0
      postconditions:
        - len(output.discounted_returns) == len(input.step_rewards)
        - len(output.advantages) == len(input.step_rewards)
      certificate: "Discounted return at t=T-1 strictly equals step_rewards[T-1]"
      compatible_adapters:
        - ADAPTER-REINFORCE-TRAINER
        - ADAPTER-LLM-RL-AGENT
      related_algos:
        - ALGO-NN-199
        - ALGO-NN-200
      references:
        - "https://doi.org/10.1007/BF00992696"
    ---
    """

    @classmethod
    def forward(
        cls,
        action_probabilities: List[float],
        step_rewards: List[float],
        baseline_values: List[float],
        gamma_discount: float = 0.99,
        normalize_advantages: bool = True,
    ) -> Dict[str, Any]:
        T = len(step_rewards)
        if T == 0:
            raise ValueError("Precondition failed: step_rewards cannot be empty")
        if len(action_probabilities) != T or len(baseline_values) != T:
            raise ValueError("Precondition failed: trajectory lengths must match")
        if any(p <= 0.0 for p in action_probabilities):
            raise ValueError("Precondition failed: action probabilities must be strictly positive")
        if not (0.0 <= gamma_discount <= 1.0):
            raise ValueError("Precondition failed: gamma_discount must be in [0, 1]")

        eps = 1e-8

        # 1. Backward accumulation of Monte Carlo discounted returns G_t
        # G_t = r_t + gamma * G_{t+1}
        returns = [0.0] * T
        running_g = 0.0
        for t in reversed(range(T)):
            running_g = step_rewards[t] + gamma_discount * running_g
            returns[t] = running_g

        # 2. Advantage calculation: A_t = G_t - b_t
        raw_advantages = [returns[t] - baseline_values[t] for t in range(T)]

        # 3. Optional batch advantage normalization (whitening)
        if normalize_advantages and T > 1:
            mean_a = sum(raw_advantages) / float(T)
            var_a = sum((a - mean_a) ** 2 for a in raw_advantages) / float(T)
            std_a = math.sqrt(var_a) + eps
            advantages = [(a - mean_a) / std_a for a in raw_advantages]
        else:
            advantages = raw_advantages

        # 4. Policy gradient loss: L = - (1/T) * sum_t ln(pi_t) * A_t
        loss_accum = 0.0
        for t in range(T):
            p = max(eps, action_probabilities[t])
            loss_accum += math.log(p) * advantages[t]
        pg_loss = - (loss_accum / float(T))

        total_return = sum(step_rewards)

        return {
            "discounted_returns": returns,
            "advantages": advantages,
            "policy_gradient_loss": pg_loss,
            "trajectory_return": total_return,
        }

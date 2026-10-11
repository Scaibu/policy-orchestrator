from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoImitationLearningDagger:
    """
    ---
    contract:
      algo_id: ALGO-NN-206
      name: NnAlgoImitationLearningDagger
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - imitation_learning
        - behavior_cloning
        - dagger
        - dataset_aggregation
      inputs:
        type: object
        required:
          - policy_predicted_actions
          - expert_actions
        properties:
          policy_predicted_actions:
            type: array
            items:
              type: number
            description: Learner policy action outputs pi_theta(s_t) on visited states of length T.
          expert_actions:
            type: array
            items:
              type: number
            description: Ground truth expert action labels pi^*(s_t) on learner visited states.
          current_iteration:
            type: integer
            default: 1
            description: DAgger aggregation round index i >= 1.
          initial_beta:
            type: number
            default: 0.5
            description: Base probability parameter beta_0 in beta_i = beta_0^(i-1).
          history_dataset_size:
            type: integer
            default: 0
            description: Total cumulative state-action pairs in replay buffer before current iteration.
          trajectory_horizon:
            type: integer
            default: 50
            description: Episode length horizon T for error bound comparison.
      outputs:
        type: object
        required:
          - beta_mixing_probability
          - iteration_imitation_loss
          - aggregated_dataset_size
          - dagger_error_bound
          - bc_error_bound
        properties:
          beta_mixing_probability:
            type: number
            description: Probability of consulting expert policy beta_i at current iteration.
          iteration_imitation_loss:
            type: number
            description: Mean squared action discrepancy on newly visited learner states.
          aggregated_dataset_size:
            type: integer
            description: New cumulative dataset size |D_i| = |D_{i-1}| + T.
          dagger_error_bound:
            type: number
            description: Theoretical asymptotic regret bound O(epsilon * T) under DAgger.
          bc_error_bound:
            type: number
            description: Compounding quadratic error bound O(epsilon * T^2) under pure Behavior Cloning.
      complexity:
        time: O(T)
        space: O(T)
      preconditions:
        - len(input.policy_predicted_actions) > 0
        - len(input.expert_actions) == len(input.policy_predicted_actions)
        - input.current_iteration >= 1
        - 0.0 <= input.initial_beta <= 1.0
        - input.history_dataset_size >= 0
        - input.trajectory_horizon >= 1
      postconditions:
        - output.iteration_imitation_loss >= 0.0
        - output.aggregated_dataset_size == input.history_dataset_size + len(input.expert_actions)
      certificate: "If policy_predicted_actions equals expert_actions, iteration_imitation_loss is 0.0"
      compatible_adapters:
        - ADAPTER-IMITATION-LEARNING-TRAINER
        - ADAPTER-DAGGER-AGGREGATOR
      related_algos:
        - ALGO-NN-201
        - ALGO-NN-205
      references:
        - "https://arxiv.org/abs/1011.0686"
    ---
    """

    @classmethod
    def forward(
        cls,
        policy_predicted_actions: List[float],
        expert_actions: List[float],
        current_iteration: int = 1,
        initial_beta: float = 0.5,
        history_dataset_size: int = 0,
        trajectory_horizon: int = 50,
    ) -> Dict[str, Any]:
        T = len(policy_predicted_actions)
        if T == 0:
            raise ValueError("Precondition failed: policy_predicted_actions must not be empty")
        if len(expert_actions) != T:
            raise ValueError("Precondition failed: expert_actions must match policy_predicted_actions length")
        if current_iteration < 1:
            raise ValueError("Precondition failed: current_iteration must be at least 1")
        if not (0.0 <= initial_beta <= 1.0):
            raise ValueError("Precondition failed: initial_beta must be in [0, 1]")
        if history_dataset_size < 0:
            raise ValueError("Precondition failed: history_dataset_size must be non-negative")
        if trajectory_horizon < 1:
            raise ValueError("Precondition failed: trajectory_horizon must be positive")

        # beta_i = beta_0^(i - 1)
        beta_i = math.pow(initial_beta, current_iteration - 1)

        # Compute empirical imitation loss (MSE between policy actions and expert demonstrations)
        sq_err = 0.0
        for t in range(T):
            diff = policy_predicted_actions[t] - expert_actions[t]
            sq_err += diff * diff

        iteration_loss = sq_err / T
        epsilon = math.sqrt(iteration_loss)

        # Regret bounds: BC is O(epsilon * T^2); DAgger is O(epsilon * T)
        H = float(trajectory_horizon)
        bc_bound = epsilon * (H * H)
        dagger_bound = 2.0 * epsilon * H

        new_dataset_size = history_dataset_size + T

        return {
            "beta_mixing_probability": beta_i,
            "iteration_imitation_loss": iteration_loss,
            "aggregated_dataset_size": new_dataset_size,
            "dagger_error_bound": dagger_bound,
            "bc_error_bound": bc_bound,
        }

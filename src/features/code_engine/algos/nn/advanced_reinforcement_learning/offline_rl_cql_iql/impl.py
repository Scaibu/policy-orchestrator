from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoOfflineRlCqlIql:
    """
    ---
    contract:
      algo_id: ALGO-NN-205
      name: NnAlgoOfflineRlCqlIql
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - reinforcement_learning
        - offline_rl
        - cql
        - iql
        - conservative_q
        - expectile_regression
      inputs:
        type: object
        required:
          - dataset_q_values
          - candidate_q_values
          - bellman_targets
          - current_v_values
        properties:
          dataset_q_values:
            type: array
            items:
              type: number
            description: Q-values evaluated on offline dataset actions Q(s, a_data) of length N.
          candidate_q_values:
            type: array
            items:
              type: array
              items:
                type: number
            description: Q-values evaluated on discrete action sets or OOD samples for each transition.
          bellman_targets:
            type: array
            items:
              type: number
            description: Standard TD Bellman targets r + gamma * V(s') of length N.
          current_v_values:
            type: array
            items:
              type: number
            description: Current state value estimates V(s) for IQL expectile evaluation.
          cql_alpha:
            type: number
            default: 1.0
            description: Trade-off parameter weighting conservative CQL logsumexp penalty.
          iql_expectile:
            type: number
            default: 0.7
            description: Asymmetric expectile parameter tau in (0, 1) for upper-tail value estimation.
          iql_temperature:
            type: number
            default: 3.0
            description: Inverse temperature beta for advantage-weighted regression policy weights.
      outputs:
        type: object
        required:
          - bellman_loss
          - cql_penalty
          - cql_critic_loss
          - iql_value_loss
          - iql_policy_weights
        properties:
          bellman_loss:
            type: number
            description: Standard MSE Bellman error on dataset actions mean(Q - y)^2.
          cql_penalty:
            type: number
            description: Mean conservative gap mean(logsumexp(Q) - Q_data).
          cql_critic_loss:
            type: number
            description: Total CQL objective: bellman_loss + cql_alpha * cql_penalty.
          iql_value_loss:
            type: number
            description: Mean expectile loss |tau - I(u < 0)| * u^2 where u = Q - V.
          iql_policy_weights:
            type: array
            items:
              type: number
            description: Exponentiated advantages exp(beta * (Q - V)) clamped for numerical stability.
      complexity:
        time: O(N * A)
        space: O(N * A)
      preconditions:
        - len(input.dataset_q_values) > 0
        - len(input.candidate_q_values) == len(input.dataset_q_values)
        - len(input.bellman_targets) == len(input.dataset_q_values)
        - len(input.current_v_values) == len(input.dataset_q_values)
        - 0.0 < input.iql_expectile < 1.0
        - input.cql_alpha >= 0.0
        - input.iql_temperature > 0.0
      postconditions:
        - len(output.iql_policy_weights) == len(input.dataset_q_values)
        - output.bellman_loss >= 0.0
        - output.iql_value_loss >= 0.0
      certificate: "If dataset_q_values equal bellman_targets and candidate_q_values contain only dataset_q, cql_penalty is 0.0"
      compatible_adapters:
        - ADAPTER-OFFLINE-RL-TRAINER
        - ADAPTER-CONSERVATIVE-CRITIC
      related_algos:
        - ALGO-NN-201
        - ALGO-NN-202
      references:
        - "https://arxiv.org/abs/2006.04779"
        - "https://arxiv.org/abs/2110.06169"
    ---
    """

    @classmethod
    def forward(
        cls,
        dataset_q_values: List[float],
        candidate_q_values: List[List[float]],
        bellman_targets: List[float],
        current_v_values: List[float],
        cql_alpha: float = 1.0,
        iql_expectile: float = 0.7,
        iql_temperature: float = 3.0,
    ) -> Dict[str, Any]:
        N = len(dataset_q_values)
        if N == 0:
            raise ValueError("Precondition failed: dataset_q_values must not be empty")
        if (
            len(candidate_q_values) != N
            or len(bellman_targets) != N
            or len(current_v_values) != N
        ):
            raise ValueError("Precondition failed: all sequence inputs must have length N")
        if not (0.0 < iql_expectile < 1.0):
            raise ValueError("Precondition failed: iql_expectile must be in (0, 1)")
        if cql_alpha < 0.0:
            raise ValueError("Precondition failed: cql_alpha must be non-negative")
        if iql_temperature <= 0.0:
            raise ValueError("Precondition failed: iql_temperature must be positive")

        bellman_sq_err = 0.0
        cql_gap_accum = 0.0
        iql_loss_accum = 0.0
        policy_weights: List[float] = []

        for i in range(N):
            q_data = dataset_q_values[i]
            y = bellman_targets[i]
            diff = q_data - y
            bellman_sq_err += diff * diff

            # LogSumExp over candidate actions
            cands = candidate_q_values[i]
            if len(cands) == 0:
                raise ValueError("Precondition failed: candidate actions for transition must not be empty")
            max_c = max(cands)
            sum_exp = sum(math.exp(c - max_c) for c in cands)
            lse = max_c + math.log(sum_exp)
            cql_gap = lse - q_data
            cql_gap_accum += cql_gap

            # IQL expectile regression on u = Q - V
            v = current_v_values[i]
            u = q_data - v
            weight = iql_expectile if u >= 0.0 else (1.0 - iql_expectile)
            iql_loss_accum += weight * (u * u)

            # AWR policy weighting: exp(beta * adv) clamped at [0, 100]
            adv = u
            scaled_adv = min(100.0, max(-100.0, iql_temperature * adv))
            policy_weights.append(math.exp(scaled_adv))

        bellman_loss = bellman_sq_err / N
        cql_penalty = cql_gap_accum / N
        cql_critic_loss = bellman_loss + cql_alpha * cql_penalty
        iql_value_loss = iql_loss_accum / N

        return {
            "bellman_loss": bellman_loss,
            "cql_penalty": cql_penalty,
            "cql_critic_loss": cql_critic_loss,
            "iql_value_loss": iql_value_loss,
            "iql_policy_weights": policy_weights,
        }

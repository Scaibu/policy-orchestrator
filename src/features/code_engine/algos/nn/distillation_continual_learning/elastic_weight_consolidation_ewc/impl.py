from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoElasticWeightConsolidationEwc:
    """
    ---
    contract:
      algo_id: ALGO-NN-227
      name: NnAlgoElasticWeightConsolidationEwc
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - continual_learning
        - catastrophic_forgetting
        - elastic_weight_consolidation
        - ewc
        - fisher_information
      inputs:
        type: object
        required:
          - current_weights
          - old_optimal_weights
          - fisher_information_diagonal
        properties:
          current_weights:
            type: array
            items:
              type: number
            description: Current model parameter values theta of dimension P.
          old_optimal_weights:
            type: array
            items:
              type: number
            description: Optimal parameter values theta*_A on previous task A of dimension P.
          fisher_information_diagonal:
            type: array
            items:
              type: number
            description: Diagonal empirical Fisher information elements F_i >= 0 representing parameter importance.
          ewc_lambda:
            type: number
            default: 100.0
            description: Regularization penalty strength hyperparameter lambda >= 0.
          task_b_loss:
            type: number
            default: 0.0
            description: Current task B objective loss without regularization.
      outputs:
        type: object
        required:
          - total_loss
          - ewc_penalty
          - task_b_loss
          - regularized_gradients
          - max_weight_drift
          - active_parameter_count
        properties:
          total_loss:
            type: number
            description: Combined objective loss L_total = L_B + (lambda / 2) * sum_i F_i * (theta_i - theta*_A,i)^2.
          ewc_penalty:
            type: number
            description: Elastic quadratic penalty value.
          task_b_loss:
            type: number
            description: Base task B loss.
          regularized_gradients:
            type: array
            items:
              type: number
            description: EWC penalty gradient vector lambda * F_i * (theta_i - theta*_A,i).
          max_weight_drift:
            type: number
            description: Maximum absolute drift |theta_i - theta*_A,i| across all parameters.
          active_parameter_count:
            type: integer
            description: Total number of evaluated parameters P.
      complexity:
        time: O(P)
        space: O(P)
      preconditions:
        - len(input.current_weights) > 0
        - len(input.current_weights) == len(input.old_optimal_weights)
        - len(input.current_weights) == len(input.fisher_information_diagonal)
        - all(f >= 0.0 for f in input.fisher_information_diagonal)
        - input.ewc_lambda >= 0.0
        - input.task_b_loss >= 0.0
      postconditions:
        - output.ewc_penalty >= 0.0
        - output.total_loss >= output.task_b_loss
        - len(output.regularized_gradients) == len(input.current_weights)
        - output.max_weight_drift >= 0.0
        - output.active_parameter_count == len(input.current_weights)
      certificate: "EWC penalty equals (lambda / 2) * sum_i F_i * (theta_i - theta*_A,i)^2"
      compatible_adapters:
        - ADAPTER-EWC-CONTINUAL-LEARNING
        - ADAPTER-FISHER-REGULARIZER
      related_algos:
        - ALGO-NN-224
        - ALGO-NN-228
      references:
        - "https://arxiv.org/abs/1612.00796"
        - "https://arxiv.org/abs/1712.03847"
    ---
    """

    @classmethod
    def forward(
        cls,
        current_weights: List[float],
        old_optimal_weights: List[float],
        fisher_information_diagonal: List[float],
        ewc_lambda: float = 100.0,
        task_b_loss: float = 0.0,
    ) -> Dict[str, Any]:
        p = len(current_weights)
        if p == 0:
            raise ValueError("Precondition failed: weights must not be empty")
        if len(old_optimal_weights) != p or len(fisher_information_diagonal) != p:
            raise ValueError("Precondition failed: vector dimensions must match")
        if any(f < 0.0 for f in fisher_information_diagonal):
            raise ValueError("Precondition failed: Fisher information must be non-negative")
        if ewc_lambda < 0.0:
            raise ValueError("Precondition failed: ewc_lambda must be non-negative")
        if task_b_loss < 0.0:
            raise ValueError("Precondition failed: task_b_loss must be non-negative")

        penalty_sum = 0.0
        max_drift = 0.0
        reg_grads: List[float] = []

        for i in range(p):
            theta_i = current_weights[i]
            theta_star_i = old_optimal_weights[i]
            f_i = fisher_information_diagonal[i]

            diff = theta_i - theta_star_i
            abs_diff = abs(diff)
            if abs_diff > max_drift:
                max_drift = abs_diff

            penalty_sum += f_i * (diff ** 2)
            reg_grads.append(ewc_lambda * f_i * diff)

        ewc_penalty = 0.5 * ewc_lambda * penalty_sum
        total_loss = task_b_loss + ewc_penalty

        return {
            "total_loss": total_loss,
            "ewc_penalty": ewc_penalty,
            "task_b_loss": task_b_loss,
            "regularized_gradients": reg_grads,
            "max_weight_drift": max_drift,
            "active_parameter_count": p,
        }

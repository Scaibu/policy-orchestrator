from __future__ import annotations

import math
from typing import Any, Dict


class NnAlgoScalingLawsComputeOptimal:
    """
    ---
    contract:
      algo_id: ALGO-NN-138
      name: NnAlgoScalingLawsComputeOptimal
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - scaling_laws
        - chinchilla
        - compute_optimal
      inputs:
        type: object
        required:
          - compute_budget_flops
        properties:
          compute_budget_flops:
            type: number
            minimum: 1.0
            description: Total training compute budget C in FLOPs.
          tokens_per_param_ratio:
            type: number
            default: 20.0
            description: Chinchilla token-to-parameter optimal ratio G.
      outputs:
        type: object
        required:
          - optimal_parameters
          - optimal_training_tokens
          - estimated_loss
        properties:
          optimal_parameters:
            type: number
            description: Compute-optimal model parameter count N*.
          optimal_training_tokens:
            type: number
            description: Compute-optimal dataset token count D*.
          estimated_loss:
            type: number
            description: Predicted test cross-entropy loss L(N*, D*).
      parameters: {}
      input_assumptions:
        - compute_budget_flops > 0.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Chinchilla power law estimation"
      uses_model: false
      complexity:
        variables:
          C: compute_budget_flops
        time_worst: O(1)
        time_typical: O(1)
        space: O(1)
      preconditions:
        - input.compute_budget_flops > 0.0
      postconditions:
        - output.optimal_parameters > 0.0
        - output.optimal_training_tokens > 0.0
      certificate: "6 * optimal_parameters * optimal_training_tokens approx equals compute_budget_flops"
      compatible_adapters:
        - ADAPTER-SCALING-PLANNER
      related_algos:
        - ALGO-NN-103
      references:
        - "https://arxiv.org/abs/2203.15556"
        - "https://arxiv.org/abs/2001.08361"
    ---
    """

    @staticmethod
    def calculate_budget(
        compute_budget_flops: float,
        tokens_per_param_ratio: float = 20.0,
    ) -> Dict[str, Any]:
        if compute_budget_flops <= 0.0 or tokens_per_param_ratio <= 0.0:
            raise ValueError("Precondition failed: positive compute budget required")

        N_star = math.sqrt(compute_budget_flops / (6.0 * tokens_per_param_ratio))
        D_star = tokens_per_param_ratio * N_star

        E = 1.69
        A = 406.4
        B = 410.7
        alpha = 0.34
        beta = 0.28
        pred_loss = E + (A / math.pow(N_star, alpha)) + (B / math.pow(D_star, beta))

        return {
            "optimal_parameters": N_star,
            "optimal_training_tokens": D_star,
            "estimated_loss": pred_loss,
        }

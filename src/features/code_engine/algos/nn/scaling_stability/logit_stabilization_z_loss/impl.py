from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLogitStabilizationZLoss:
    """
    ---
    contract:
      algo_id: ALGO-NN-140
      name: NnAlgoLogitStabilizationZLoss
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - z_loss
        - stability
        - large_scale_training
      inputs:
        type: object
        required:
          - logits
        properties:
          logits:
            type: array
            items:
              type: array
              items:
                type: number
            description: Logits matrix of shape (batch_size, vocab_size).
          alpha:
            type: number
            default: 0.0001
            description: Z-loss regularizer coefficient alpha.
      outputs:
        type: object
        required:
          - z_loss
          - max_logit
          - mean_logsumexp
        properties:
          z_loss:
            type: number
            description: Scaled auxiliary z-loss alpha * mean((log sum exp(logits))^2).
          max_logit:
            type: number
            description: Maximum observed logit magnitude.
          mean_logsumexp:
            type: number
            description: Average log-sum-exp normalization value.
      parameters: {}
      input_assumptions:
        - logits non-empty with finite values
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard numerical roundoff"
      uses_model: false
      complexity:
        variables:
          B: batch_size
          V: vocab_size
        time_worst: O(B * V)
        time_typical: O(B * V)
        space: O(B)
      preconditions:
        - len(input.logits) > 0
        - input.alpha >= 0.0
      postconditions:
        - output.z_loss >= 0.0
      certificate: "z_loss == alpha * (1/B) * sum((logsumexp_i)^2)"
      compatible_adapters:
        - ADAPTER-STABILITY-LOSS
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-126
      references:
        - "https://arxiv.org/abs/2204.02311"
        - "https://arxiv.org/abs/2309.16609"
    ---
    """

    @staticmethod
    def compute(logits: List[List[float]], alpha: float = 1e-4) -> Dict[str, Any]:
        if not logits or alpha < 0.0:
            raise ValueError("Precondition failed: invalid inputs")

        B = len(logits)
        total_z_loss = 0.0
        total_lse = 0.0
        global_max_l = -float("inf")

        for row in logits:
            max_l = max(row)
            global_max_l = max(global_max_l, max_l)
            exp_sum = sum(math.exp(z - max_l) for z in row)
            lse = max_l + math.log(exp_sum)
            total_lse += lse
            total_z_loss += lse * lse

        mean_lse = total_lse / float(B)
        aux_z_loss = alpha * (total_z_loss / float(B))

        return {
            "z_loss": aux_z_loss,
            "max_logit": global_max_l,
            "mean_logsumexp": mean_lse,
        }

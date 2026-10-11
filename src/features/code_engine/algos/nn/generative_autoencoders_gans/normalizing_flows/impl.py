from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoNormalizingFlows:
    """
    ---
    contract:
      algo_id: ALGO-NN-158
      name: NnAlgoNormalizingFlows
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - normalizing_flow
        - realnvp
        - exact_likelihood
        - invertible_mapping
      inputs:
        type: object
        required:
          - input_vector
          - scale_weights
          - scale_bias
          - trans_weights
          - trans_bias
          - split_dim
        properties:
          input_vector:
            type: array
            items:
              type: number
            description: Input vector z or x of dimension D.
          scale_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Scale transformation network weights of shape (D - split_dim, split_dim).
          scale_bias:
            type: array
            items:
              type: number
            description: Scale transformation network bias of dimension D - split_dim.
          trans_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Translation transformation network weights of shape (D - split_dim, split_dim).
          trans_bias:
            type: array
            items:
              type: number
            description: Translation transformation network bias of dimension D - split_dim.
          split_dim:
            type: integer
            minimum: 1
            description: Dimension d defining the partition boundary [0, d) and [d, D).
          inverse:
            type: boolean
            default: false
            description: Forward mapping (sampling) if false; inverse mapping (density evaluation) if true.
      outputs:
        type: object
        required:
          - output_vector
          - log_det_jacobian
          - log_likelihood
        properties:
          output_vector:
            type: array
            items:
              type: number
            description: Transformed vector in target domain of dimension D.
          log_det_jacobian:
            type: number
            description: Sum of log-scale factors representing log|det(J)|.
          log_likelihood:
            type: number
            description: Exact log-density evaluation under standard Gaussian base measure.
      parameters: {}
      input_assumptions:
        - input_vector dimension D > split_dim >= 1
        - scale_weights and trans_weights have matching valid shapes
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: exact_invertible
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Inversion is exact within machine epsilon"
      uses_model: false
      complexity:
        variables:
          D: total vector dimension
          d: split dimension
        time_worst: O((D - d) * d)
        time_typical: O((D - d) * d)
        space: O(D)
      preconditions:
        - len(input.input_vector) > input.split_dim >= 1
        - len(input.scale_weights) == len(input.input_vector) - input.split_dim
        - len(input.scale_weights[0]) == input.split_dim
        - len(input.trans_weights) == len(input.scale_weights)
        - len(input.scale_bias) == len(input.scale_weights)
        - len(input.trans_bias) == len(input.scale_weights)
      postconditions:
        - len(output.output_vector) == len(input.input_vector)
      certificate: "Forward then inverse yields original vector within numerical tolerance"
      compatible_adapters:
        - ADAPTER-FLOW-ESTIMATOR
        - ADAPTER-INVERTIBLE-SAMPLER
      related_algos:
        - ALGO-NN-152
        - ALGO-NN-168
      references:
        - "https://arxiv.org/abs/1605.08803"
    ---
    """

    @staticmethod
    def forward(
        input_vector: List[float],
        scale_weights: List[List[float]],
        scale_bias: List[float],
        trans_weights: List[List[float]],
        trans_bias: List[float],
        split_dim: int,
        inverse: bool = False,
    ) -> Dict[str, Any]:
        D = len(input_vector)
        d = split_dim
        if d < 1 or d >= D:
            raise ValueError("Precondition failed: split_dim must be strictly between 1 and D-1")

        D_rem = D - d
        if len(scale_weights) != D_rem or any(len(row) != d for row in scale_weights):
            raise ValueError("Precondition failed: scale_weights dimension mismatch")
        if len(trans_weights) != D_rem or any(len(row) != d for row in trans_weights):
            raise ValueError("Precondition failed: trans_weights dimension mismatch")
        if len(scale_bias) != D_rem or len(trans_bias) != D_rem:
            raise ValueError("Precondition failed: bias dimension mismatch")

        # Identity partition
        z_1 = input_vector[:d]
        z_2 = input_vector[d:]

        # Compute scale s and translation t from z_1
        s_vals: List[float] = []
        t_vals: List[float] = []
        for i in range(D_rem):
            s_act = scale_bias[i]
            t_act = trans_bias[i]
            for j in range(d):
                s_act += scale_weights[i][j] * z_1[j]
                t_act += trans_weights[i][j] * z_1[j]
            # Use tanh clamping on scale for numerical stability: s in [-2, 2]
            s_clamped = 2.0 * math.tanh(s_act * 0.5)
            s_vals.append(s_clamped)
            t_vals.append(t_act)

        output_vector: List[float] = list(z_1)
        log_det_j = sum(s_vals)

        if not inverse:
            # Forward: x_2 = z_2 * exp(s) + t
            for i in range(D_rem):
                val = z_2[i] * math.exp(s_vals[i]) + t_vals[i]
                output_vector.append(val)
        else:
            # Inverse: z_2 = (x_2 - t) * exp(-s)
            for i in range(D_rem):
                val = (z_2[i] - t_vals[i]) * math.exp(-s_vals[i])
                output_vector.append(val)
            log_det_j = -log_det_j

        # Base density under standard Gaussian: log p(z) = -0.5 * (D * log(2pi) + sum(z_i^2))
        z_eval = input_vector if not inverse else output_vector
        sum_sq = sum(v * v for v in z_eval)
        base_log_prob = -0.5 * (float(D) * math.log(2.0 * math.pi) + sum_sq)
        log_likelihood = base_log_prob - log_det_j

        return {
            "output_vector": output_vector,
            "log_det_jacobian": log_det_j,
            "log_likelihood": log_likelihood,
        }

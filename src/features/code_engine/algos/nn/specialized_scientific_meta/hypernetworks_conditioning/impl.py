from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoHypernetworksConditioning:
    """
    ---
    contract:
      algo_id: ALGO-NN-184
      name: NnAlgoHypernetworksConditioning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - hypernetwork
        - dynamic_weights
        - low_rank_generation
        - conditioned_adaptation
      inputs:
        type: object
        required:
          - input_vector_x
          - condition_c
          - base_weights
          - hyper_weights_a
          - hyper_weights_b
          - rank_r
        properties:
          input_vector_x:
            type: array
            items:
              type: number
            description: Primary network activation vector x of dimension D_in.
          condition_c:
            type: array
            items:
              type: number
            description: Context/task conditioning vector c of dimension d_c.
          base_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Static base weight matrix W_0 of shape (D_out, D_in).
          hyper_weights_a:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hypernetwork projection for factor A of shape (D_out * rank_r, d_c).
          hyper_weights_b:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hypernetwork projection for factor B of shape (D_in * rank_r, d_c).
          rank_r:
            type: integer
            minimum: 1
            description: Bottleneck rank r for low-rank weight generation.
          scale_alpha:
            type: number
            default: 1.0
            description: Scaling multiplier for generated weight delta.
      outputs:
        type: object
        required:
          - output_vector_y
          - effective_weight_norm
          - delta_weight_norm
        properties:
          output_vector_y:
            type: array
            items:
              type: number
            description: Evaluated primary output y = (W_0 + alpha * A * B^T / r) * x of dimension D_out.
          effective_weight_norm:
            type: number
            description: Frobenius norm of the synthesized effective weight matrix W_eff.
          delta_weight_norm:
            type: number
            description: Frobenius norm of the dynamic weight update delta W.
      parameters: {}
      input_assumptions:
        - base_weights has shape (D_out, D_in) with D_out >= 1, D_in >= 1
        - input_vector_x has length D_in
        - condition_c has length d_c >= 1
        - hyper_weights_a has shape (D_out * r, d_c), hyper_weights_b has shape (D_in * r, d_c)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact low-rank outer product synthesis"
      uses_model: false
      complexity:
        variables:
          D_in: input dimension
          D_out: output dimension
          d_c: condition dimension
          r: rank
        time_worst: O(D_out * r * d_c + D_in * r * d_c + D_out * D_in * r)
        time_typical: O(D_out * r * d_c + D_in * r * d_c + D_out * D_in * r)
        space: O(D_out * D_in)
      preconditions:
        - len(input.base_weights) > 0 and len(input.base_weights[0]) > 0
        - len(input.input_vector_x) == len(input.base_weights[0])
        - len(input.condition_c) > 0
        - input.rank_r >= 1
      postconditions:
        - len(output.output_vector_y) == len(input.base_weights)
        - output.effective_weight_norm >= 0.0
        - output.delta_weight_norm >= 0.0
      certificate: "Effective weights satisfy W_0 + (alpha / r) * A * B^T"
      compatible_adapters:
        - ADAPTER-HYPERNETWORK-GENERATOR
        - ADAPTER-DYNAMIC-LORA-SYNTHESIZER
      related_algos:
        - ALGO-NN-170
        - ALGO-NN-178
      references:
        - "https://arxiv.org/abs/1609.09106"
    ---
    """

    @classmethod
    def forward(
        cls,
        input_vector_x: List[float],
        condition_c: List[float],
        base_weights: List[List[float]],
        hyper_weights_a: List[List[float]],
        hyper_weights_b: List[List[float]],
        rank_r: int,
        scale_alpha: float = 1.0,
    ) -> Dict[str, Any]:
        D_out = len(base_weights)
        if D_out == 0 or len(base_weights[0]) == 0:
            raise ValueError("Precondition failed: base_weights cannot be empty")
        D_in = len(base_weights[0])
        d_c = len(condition_c)

        if len(input_vector_x) != D_in or d_c == 0 or rank_r < 1:
            raise ValueError("Precondition failed: dimension mismatch or invalid rank")
        if len(hyper_weights_a) != D_out * rank_r or any(len(r) != d_c for r in hyper_weights_a):
            raise ValueError("Precondition failed: hyper_weights_a dimension mismatch")
        if len(hyper_weights_b) != D_in * rank_r or any(len(r) != d_c for r in hyper_weights_b):
            raise ValueError("Precondition failed: hyper_weights_b dimension mismatch")

        # 1. Project condition to low-rank matrices A in (D_out, r) and B in (D_in, r)
        flat_a = []
        for i in range(D_out * rank_r):
            val = sum(hyper_weights_a[i][k] * condition_c[k] for k in range(d_c))
            flat_a.append(val)

        flat_b = []
        for j in range(D_in * rank_r):
            val = sum(hyper_weights_b[j][k] * condition_c[k] for k in range(d_c))
            flat_b.append(val)

        # 2. Reconstruct delta weight matrix: Delta_W = (scale / rank) * A * B^T
        scale_factor = scale_alpha / float(rank_r)
        delta_norm_sq = 0.0
        eff_norm_sq = 0.0

        w_eff: List[List[float]] = []
        for i in range(D_out):
            row_eff: List[float] = []
            for j in range(D_in):
                # Dot product of A[i, :] and B[j, :] across r
                delta_val = 0.0
                for r in range(rank_r):
                    a_elem = flat_a[i * rank_r + r]
                    b_elem = flat_b[j * rank_r + r]
                    delta_val += a_elem * b_elem

                delta_val *= scale_factor
                delta_norm_sq += delta_val * delta_val

                total_w = base_weights[i][j] + delta_val
                eff_norm_sq += total_w * total_w
                row_eff.append(total_w)
            w_eff.append(row_eff)

        # 3. Apply effective weights to input: y = W_eff * x
        output_y: List[float] = []
        for i in range(D_out):
            val = sum(w_eff[i][j] * input_vector_x[j] for j in range(D_in))
            output_y.append(val)

        return {
            "output_vector_y": output_y,
            "effective_weight_norm": math.sqrt(eff_norm_sq),
            "delta_weight_norm": math.sqrt(delta_norm_sq),
        }

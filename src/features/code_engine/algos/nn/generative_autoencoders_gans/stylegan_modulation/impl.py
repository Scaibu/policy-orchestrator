from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoStyleganModulation:
    """
    ---
    contract:
      algo_id: ALGO-NN-156
      name: NnAlgoStyleganModulation
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - stylegan
        - weight_demodulation
        - style_conditioning
      inputs:
        type: object
        required:
          - style_vector
          - affine_weights
          - affine_bias
          - conv_weights
        properties:
          style_vector:
            type: array
            items:
              type: number
            description: Disentangled intermediate latent vector w of dimension d_w.
          affine_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Affine style transformation matrix A of shape (C_in, d_w).
          affine_bias:
            type: array
            items:
              type: number
            description: Affine style bias vector b_A of dimension C_in.
          conv_weights:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Base convolution weight tensor W of shape (C_out, C_in, K).
          eps:
            type: number
            default: 1e-8
            description: Numerical stability denominator epsilon.
      outputs:
        type: object
        required:
          - style_scales
          - demodulated_weights
          - channel_norms
        properties:
          style_scales:
            type: array
            items:
              type: number
            description: Injected per-input-channel scale factors s_j.
          demodulated_weights:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Style-demodulated convolution weights W_prime of shape (C_out, C_in, K).
          channel_norms:
            type: array
            items:
              type: number
            description: Output channel normalization factors sigma_i.
      parameters: {}
      input_assumptions:
        - style_vector has positive dimension d_w
        - conv_weights is non-empty 3D tensor of shape (C_out, C_in, K)
        - affine_weights has shape (C_in, d_w)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically exact within IEEE-754 precision"
      uses_model: false
      complexity:
        variables:
          C_out: output channels
          C_in: input channels
          K: spatial kernel elements
          d_w: latent style dimension
        time_worst: O(C_in * d_w + C_out * C_in * K)
        time_typical: O(C_in * d_w + C_out * C_in * K)
        space: O(C_out * C_in * K)
      preconditions:
        - len(input.style_vector) > 0
        - len(input.affine_weights) > 0 and len(input.affine_weights[0]) == len(input.style_vector)
        - len(input.affine_bias) == len(input.affine_weights)
        - len(input.conv_weights) > 0 and len(input.conv_weights[0]) == len(input.affine_weights)
      postconditions:
        - len(output.style_scales) == len(input.affine_weights)
        - len(output.demodulated_weights) == len(input.conv_weights)
        - len(output.channel_norms) == len(input.conv_weights)
      certificate: "Output weights are scaled and normalized per output channel"
      compatible_adapters:
        - ADAPTER-STYLEGAN-GENERATOR
      related_algos:
        - ALGO-NN-154
        - ALGO-NN-155
      references:
        - "https://arxiv.org/abs/1912.04958"
    ---
    """

    @staticmethod
    def forward(
        style_vector: List[float],
        affine_weights: List[List[float]],
        affine_bias: List[float],
        conv_weights: List[List[List[float]]],
        eps: float = 1e-8,
    ) -> Dict[str, Any]:
        d_w = len(style_vector)
        C_in = len(affine_bias)
        if d_w == 0 or C_in == 0:
            raise ValueError("Precondition failed: empty style or channels")

        if len(affine_weights) != C_in or any(len(row) != d_w for row in affine_weights):
            raise ValueError("Precondition failed: affine_weights dimension mismatch")

        C_out = len(conv_weights)
        if C_out == 0 or len(conv_weights[0]) != C_in:
            raise ValueError("Precondition failed: conv_weights channel mismatch")

        K = len(conv_weights[0][0])
        if K == 0:
            raise ValueError("Precondition failed: conv_weights kernel must be non-empty")

        # 1. Compute style scales: s_j = sum(A_{j, d} * w_d) + b_{A, j}
        # In StyleGAN2, default is 1.0 + affine transform
        style_scales: List[float] = []
        for j in range(C_in):
            val = affine_bias[j] + 1.0
            for d in range(d_w):
                val += affine_weights[j][d] * style_vector[d]
            style_scales.append(val)

        # 2. Modulate weights: W'_{i, j, k} = s_j * W_{i, j, k}
        # 3. Demodulate: W''_{i, j, k} = W'_{i, j, k} / sqrt(sum_{j, k} (W'_{i, j, k})^2 + eps)
        demodulated_weights: List[List[List[float]]] = []
        channel_norms: List[float] = []

        for i in range(C_out):
            # Compute norm factor for output channel i
            sum_sq = 0.0
            for j in range(C_in):
                s_j = style_scales[j]
                for k in range(K):
                    w_mod = s_j * conv_weights[i][j][k]
                    sum_sq += w_mod * w_mod

            norm = math.sqrt(sum_sq + eps)
            channel_norms.append(norm)

            out_matrix: List[List[float]] = []
            for j in range(C_in):
                s_j = style_scales[j]
                kernel_row: List[float] = []
                for k in range(K):
                    w_mod = s_j * conv_weights[i][j][k]
                    kernel_row.append(w_mod / norm)
                out_matrix.append(kernel_row)
            demodulated_weights.append(out_matrix)

        return {
            "style_scales": style_scales,
            "demodulated_weights": demodulated_weights,
            "channel_norms": channel_norms,
        }

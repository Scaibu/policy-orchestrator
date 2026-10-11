from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoDiffusionTransformersDit:
    """
    ---
    contract:
      algo_id: ALGO-NN-167
      name: NnAlgoDiffusionTransformersDit
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - dit
        - diffusion_transformer
        - adaln_zero
        - conditioning_gate
      inputs:
        type: object
        required:
          - token_activations
          - gamma
          - beta
          - alpha_gate
        properties:
          token_activations:
            type: array
            items:
              type: array
              items:
                type: number
            description: Input sequence tokens of shape (N_tokens, d_model).
          gamma:
            type: array
            items:
              type: number
            description: Adaptive scale parameter gamma(c) of dimension d_model.
          beta:
            type: array
            items:
              type: number
            description: Adaptive shift parameter beta(c) of dimension d_model.
          alpha_gate:
            type: array
            items:
              type: number
            description: Residual modulation gate alpha(c) of dimension d_model (zero-initialized).
          sublayer_outputs:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention or MLP sublayer outputs of shape (N_tokens, d_model).
      outputs:
        type: object
        required:
          - normalized_modulated_tokens
          - gated_residual_output
          - gate_norm
        properties:
          normalized_modulated_tokens:
            type: array
            items:
              type: array
              items:
                type: number
            description: Modulated normalized states LN(x) * (1 + gamma) + beta.
          gated_residual_output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Residual output x + alpha * sublayer_output.
          gate_norm:
            type: number
            description: L2 norm of the adaptive gating parameter alpha.
      parameters: {}
      input_assumptions:
        - token_activations has uniform dimension d_model matching gamma, beta, and alpha_gate
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically stabilized LayerNorm with epsilon 1e-6"
      uses_model: false
      complexity:
        variables:
          N: number of tokens
          D: model embedding dimension
        time_worst: O(N * D)
        time_typical: O(N * D)
        space: O(N * D)
      preconditions:
        - len(input.token_activations) > 0 and len(input.token_activations[0]) > 0
        - len(input.gamma) == len(input.token_activations[0])
        - len(input.beta) == len(input.token_activations[0])
        - len(input.alpha_gate) == len(input.token_activations[0])
      postconditions:
        - len(output.normalized_modulated_tokens) == len(input.token_activations)
        - len(output.gated_residual_output) == len(input.token_activations)
        - output.gate_norm >= 0.0
      certificate: "When alpha_gate is zero, gated_residual_output strictly equals token_activations"
      compatible_adapters:
        - ADAPTER-DIT-BLOCK
        - ADAPTER-ADALN-ZERO
      related_algos:
        - ALGO-NN-128
        - ALGO-NN-161
        - ALGO-NN-166
      references:
        - "https://arxiv.org/abs/2212.09748"
    ---
    """

    @staticmethod
    def forward(
        token_activations: List[List[float]],
        gamma: List[float],
        beta: List[float],
        alpha_gate: List[float],
        sublayer_outputs: List[List[float]],
    ) -> Dict[str, Any]:
        N = len(token_activations)
        if N == 0 or len(token_activations[0]) == 0:
            raise ValueError("Precondition failed: token_activations cannot be empty")
        D = len(token_activations[0])

        if len(gamma) != D or len(beta) != D or len(alpha_gate) != D:
            raise ValueError("Precondition failed: modulation parameter dimension mismatch")
        if len(sublayer_outputs) != N or any(len(r) != D for r in sublayer_outputs):
            raise ValueError("Precondition failed: sublayer_outputs dimension mismatch")

        eps = 1e-6

        # 1. Apply LayerNorm + adaLN modulation: LN(x) * (1 + gamma) + beta
        mod_tokens: List[List[float]] = []
        for i in range(N):
            row = token_activations[i]
            mean = sum(row) / float(D)
            var = sum((x - mean) ** 2 for x in row) / float(D)
            std = math.sqrt(var + eps)

            mod_row: List[float] = []
            for j in range(D):
                x_norm = (row[j] - mean) / std
                val = x_norm * (1.0 + gamma[j]) + beta[j]
                mod_row.append(val)
            mod_tokens.append(mod_row)

        # 2. Apply adaLN-Zero gated residual: y = x + alpha * sublayer_output
        gated_output: List[List[float]] = []
        for i in range(N):
            out_row: List[float] = []
            for j in range(D):
                val = token_activations[i][j] + alpha_gate[j] * sublayer_outputs[i][j]
                out_row.append(val)
            gated_output.append(out_row)

        gate_norm = math.sqrt(sum(a * a for a in alpha_gate))

        return {
            "normalized_modulated_tokens": mod_tokens,
            "gated_residual_output": gated_output,
            "gate_norm": gate_norm,
        }

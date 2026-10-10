from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.features.code_engine.algos.nn.attention_objectives.multi_head_attention.impl import (
    NnAlgoMultiHeadAttention,
)


class NnAlgoTransformerBlock:
    """
    ---
    contract:
      algo_id: ALGO-NN-112
      name: NnAlgoTransformerBlock
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - transformer_block
        - pre_norm
        - ffn
      inputs:
        type: object
        required:
          - hidden_states
          - num_heads
          - d_ff
        properties:
          hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Input sequence hidden states of shape (seq_len, d_model).
          num_heads:
            type: integer
            minimum: 1
            description: Number of attention heads.
          d_ff:
            type: integer
            minimum: 1
            description: FFN intermediate dimension.
          mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Optional attention mask.
      outputs:
        type: object
        required:
          - output_states
          - attn_residual
          - ffn_residual
        properties:
          output_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Output hidden states of shape (seq_len, d_model).
          attn_residual:
            type: array
            items:
              type: array
              items:
                type: number
            description: State after attention residual add.
          ffn_residual:
            type: array
            items:
              type: array
              items:
                type: number
            description: Final state after FFN residual add.
      parameters: {}
      input_assumptions:
        - hidden_states is non-empty 2D array
        - d_model divisible by num_heads
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point error"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          d: d_model
          d_ff: d_ff
        time_worst: O(N^2 * d + N * d * d_ff)
        time_typical: O(N^2 * d + N * d * d_ff)
        space: O(N * d)
      preconditions:
        - len(input.hidden_states) > 0
        - len(input.hidden_states[0]) % input.num_heads == 0
      postconditions:
        - len(output.output_states) == len(input.hidden_states)
        - len(output.output_states[0]) == len(input.hidden_states[0])
      certificate: "Output preserves hidden_states tensor shape"
      compatible_adapters:
        - ADAPTER-TRANSFORMER-LAYER
      related_algos:
        - ALGO-NN-102
      references:
        - "https://arxiv.org/abs/1706.03762"
        - "https://arxiv.org/abs/2002.04745"
    ---
    """

    @staticmethod
    def forward(
        hidden_states: List[List[float]],
        num_heads: int,
        d_ff: int,
        mask: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if not hidden_states:
            raise ValueError("Precondition failed: hidden_states must be non-empty")

        N = len(hidden_states)
        d = len(hidden_states[0])
        if d % num_heads != 0:
            raise ValueError("Precondition failed: d_model must be divisible by num_heads")

        def rms_norm(x_seq: List[List[float]]) -> List[List[float]]:
            normed: List[List[float]] = []
            for row in x_seq:
                ms = sum(v * v for v in row) / float(len(row))
                scale = 1.0 / math.sqrt(ms + 1e-6)
                normed.append([v * scale for v in row])
            return normed

        norm1 = rms_norm(hidden_states)
        attn_res = NnAlgoMultiHeadAttention.forward(norm1, norm1, norm1, num_heads=num_heads, mask=mask)
        attn_out = attn_res["output"]

        x_mid: List[List[float]] = []
        for i in range(N):
            x_mid.append([hidden_states[i][j] + attn_out[i][j] for j in range(d)])

        norm2 = rms_norm(x_mid)
        ffn_out: List[List[float]] = []
        for i in range(N):
            row = norm2[i]
            proj1 = [max(0.0, sum(row[j] * 0.01 for j in range(d))) for _ in range(d_ff)]
            proj2 = [sum(proj1[k] * 0.01 for k in range(d_ff)) for _ in range(d)]
            ffn_out.append(proj2)

        x_final: List[List[float]] = []
        for i in range(N):
            x_final.append([x_mid[i][j] + ffn_out[i][j] for j in range(d)])

        return {
            "output_states": x_final,
            "attn_residual": x_mid,
            "ffn_residual": x_final,
        }

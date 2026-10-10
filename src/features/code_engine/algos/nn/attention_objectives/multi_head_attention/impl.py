from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoMultiHeadAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-102
      name: NnAlgoMultiHeadAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - multi_head_attention
        - transformer
      inputs:
        type: object
        required:
          - query
          - key
          - value
          - num_heads
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query matrix of shape (N_q, d_model).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix of shape (N_k, d_model).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix of shape (N_k, d_model).
          num_heads:
            type: integer
            minimum: 1
            description: Number of heads h.
          mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Optional mask of shape (N_q, N_k).
      outputs:
        type: object
        required:
          - output
          - head_outputs
          - head_dim
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Output tensor of shape (N_q, d_model).
          head_outputs:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Per-head outputs of shape (h, N_q, d_k).
          head_dim:
            type: integer
            description: Head dimension d_k.
      parameters: {}
      input_assumptions:
        - d_model is divisible by num_heads
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point bounds"
      uses_model: false
      complexity:
        variables:
          N_q: query sequence length
          N_k: key sequence length
          d_model: model dimension
          h: number of heads
        time_worst: O(N_q * N_k * d_model)
        time_typical: O(N_q * N_k * d_model)
        space: O(h * N_q * N_k + N_q * d_model)
      preconditions:
        - len(input.query) > 0 and len(input.query[0]) % input.num_heads == 0
      postconditions:
        - len(output.output) == len(input.query)
        - len(output.output[0]) == len(input.query[0])
      certificate: "Output dimension equals input dimension"
      compatible_adapters:
        - ADAPTER-MHA
      related_algos:
        - ALGO-NN-101
      references:
        - "https://arxiv.org/abs/1706.03762"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        num_heads: int,
        mask: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        if not query or not key or not value:
            raise ValueError("Precondition failed: query, key, value must be non-empty")
        if not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError("Precondition failed: num_heads must be positive integer")

        N_q = len(query)
        d_model = len(query[0])
        if d_model % num_heads != 0:
            raise ValueError("Precondition failed: d_model must be divisible by num_heads")

        d_k = d_model // num_heads

        head_outputs: List[List[List[float]]] = []
        for h in range(num_heads):
            s = h * d_k
            e = s + d_k
            q_h = [[row[c] for c in range(s, e)] for row in query]
            k_h = [[row[c] for c in range(s, e)] for row in key]
            v_h = [[row[c] for c in range(s, e)] for row in value]

            res = NnAlgoScaledDotProductAttention.forward(q_h, k_h, v_h, mask=mask)
            head_outputs.append(res["output"])

        output: List[List[float]] = []
        for i in range(N_q):
            row: List[float] = []
            for h in range(num_heads):
                row.extend(head_outputs[h][i])
            output.append(row)

        return {
            "output": output,
            "head_outputs": head_outputs,
            "head_dim": d_k,
        }

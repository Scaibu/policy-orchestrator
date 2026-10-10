from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoGroupedQueryAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-114
      name: NnAlgoGroupedQueryAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - gqa
        - mqa
        - kv_cache_compression
      inputs:
        type: object
        required:
          - query
          - key
          - value
          - num_q_heads
          - num_kv_heads
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query sequence of shape (seq_len, num_q_heads * head_dim).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key sequence of shape (seq_len, num_kv_heads * head_dim).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value sequence of shape (seq_len, num_kv_heads * head_dim).
          num_q_heads:
            type: integer
            minimum: 1
            description: Number of query heads.
          num_kv_heads:
            type: integer
            minimum: 1
            description: Number of key/value heads (must divide num_q_heads).
      outputs:
        type: object
        required:
          - output
          - compression_ratio
          - head_dim
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention output matrix of shape (seq_len, num_q_heads * head_dim).
          compression_ratio:
            type: number
            description: KV cache memory compression factor (num_q_heads / num_kv_heads).
          head_dim:
            type: integer
            description: Dimension per head.
      parameters: {}
      input_assumptions:
        - num_q_heads is divisible by num_kv_heads
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point precision"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          H_q: num_q_heads
          H_kv: num_kv_heads
          d: head_dim
        time_worst: O(H_q * N^2 * d)
        time_typical: O(H_q * N^2 * d)
        space: O(H_q * N^2)
      preconditions:
        - input.num_q_heads % input.num_kv_heads == 0
        - len(input.query) > 0 and len(input.key) > 0 and len(input.value) > 0
      postconditions:
        - len(output.output) == len(input.query)
        - len(output.output[0]) == len(input.query[0])
      certificate: "compression_ratio == num_q_heads / num_kv_heads"
      compatible_adapters:
        - ADAPTER-GQA
      related_algos:
        - ALGO-NN-102
        - ALGO-NN-115
      references:
        - "https://arxiv.org/abs/2305.13245"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        num_q_heads: int,
        num_kv_heads: int,
    ) -> Dict[str, Any]:
        if not query or not key or not value:
            raise ValueError("Precondition failed: non-empty inputs required")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("Precondition failed: num_q_heads must be divisible by num_kv_heads")

        N_q = len(query)
        N_k = len(key)
        head_dim = len(query[0]) // num_q_heads
        if len(key[0]) != num_kv_heads * head_dim or len(value[0]) != num_kv_heads * head_dim:
            raise ValueError("Precondition failed: key/value head dimension mismatch")

        group_size = num_q_heads // num_kv_heads
        q_heads_out: List[List[List[float]]] = []

        for q_h in range(num_q_heads):
            kv_h = q_h // group_size

            q_s = q_h * head_dim
            q_e = q_s + head_dim
            kv_s = kv_h * head_dim
            kv_e = kv_s + head_dim

            q_vecs = [[row[c] for c in range(q_s, q_e)] for row in query]
            k_vecs = [[row[c] for c in range(kv_s, kv_e)] for row in key]
            v_vecs = [[row[c] for c in range(kv_s, kv_e)] for row in value]

            res = NnAlgoScaledDotProductAttention.forward(q_vecs, k_vecs, v_vecs)
            q_heads_out.append(res["output"])

        combined: List[List[float]] = []
        for i in range(N_q):
            row: List[float] = []
            for q_h in range(num_q_heads):
                row.extend(q_heads_out[q_h][i])
            combined.append(row)

        return {
            "output": combined,
            "compression_ratio": float(group_size),
            "head_dim": head_dim,
        }

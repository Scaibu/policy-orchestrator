from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLinearAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-121
      name: NnAlgoLinearAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - linear_attention
        - kernel_feature_map
        - recurrent_attention
      inputs:
        type: object
        required:
          - query
          - key
          - value
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query matrix Q of shape (seq_len, d).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix K of shape (seq_len, d).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix V of shape (seq_len, d_v).
          eps:
            type: number
            default: 1e-6
            description: Numerical normalizer epsilon.
      outputs:
        type: object
        required:
          - output
          - complexity_advantage
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Linear attention output of shape (seq_len, d_v).
          complexity_advantage:
            type: string
            description: Asymptotic complexity compared to standard quadratic attention.
      parameters: {}
      input_assumptions:
        - all inputs non-empty with matching lengths
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
          N: seq_len
          d: feature dimension
          d_v: value dimension
        time_worst: O(N * d * d_v)
        time_typical: O(N * d * d_v)
        space: O(d * d_v + N * d_v)
      preconditions:
        - len(input.query) > 0 and len(input.query) == len(input.key)
      postconditions:
        - len(output.output) == len(input.query)
        - len(output.output[0]) == len(input.value[0])
      certificate: "Complexity linear in sequence length O(N * d * d_v)"
      compatible_adapters:
        - ADAPTER-LINEAR-ATTN
      related_algos:
        - ALGO-NN-101
        - ALGO-NN-124
      references:
        - "https://arxiv.org/abs/2006.16236"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        eps: float = 1e-6,
    ) -> Dict[str, Any]:
        if not query or len(query) != len(key) or len(query) != len(value):
            raise ValueError("Precondition failed: inputs must be non-empty matching sequences")

        N = len(query)
        d = len(query[0])
        d_v = len(value[0])

        def phi(x_vec: List[float]) -> List[float]:
            return [math.log(1.0 + math.exp(v)) + 1.0 for v in x_vec]

        state_kv = [[0.0] * d_v for _ in range(d)]
        state_z = [0.0] * d
        output: List[List[float]] = []

        for t in range(N):
            q_t = phi(query[t])
            k_t = phi(key[t])
            v_t = value[t]

            for i in range(d):
                state_z[i] += k_t[i]
                for j in range(d_v):
                    state_kv[i][j] += k_t[i] * v_t[j]

            denom = sum(q_t[i] * state_z[i] for i in range(d)) + eps
            out_row = [0.0] * d_v
            for j in range(d_v):
                num = sum(q_t[i] * state_kv[i][j] for i in range(d))
                out_row[j] = num / denom
            output.append(out_row)

        return {
            "output": output,
            "complexity_advantage": "O(N * d * d_v) linear time vs O(N^2 * d)",
        }

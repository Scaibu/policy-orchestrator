from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRwkvRetnetRecurrent:
    """
    ---
    contract:
      algo_id: ALGO-NN-124
      name: NnAlgoRwkvRetnetRecurrent
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - retnet
        - rwkv
        - retention
        - recurrent_transformer
      inputs:
        type: object
        required:
          - query
          - key
          - value
          - gamma
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query matrix of shape (seq_len, d).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix of shape (seq_len, d).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix of shape (seq_len, d_v).
          gamma:
            type: number
            default: 0.9
            description: Retention decay factor gamma in (0, 1).
      outputs:
        type: object
        required:
          - output
          - final_recurrent_state
          - gamma
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Retention output matrix of shape (seq_len, d_v).
          final_recurrent_state:
            type: array
            items:
              type: array
              items:
                type: number
            description: Recurrent memory state S_L of shape (d, d_v).
          gamma:
            type: number
            description: Applied retention decay.
      parameters: {}
      input_assumptions:
        - 0.0 < gamma < 1.0
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact recurrent decay propagation"
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
        - 0.0 < input.gamma < 1.0
      postconditions:
        - len(output.output) == len(input.query)
      certificate: "Exact O(1) state recurrence per step"
      compatible_adapters:
        - ADAPTER-RETNET
      related_algos:
        - ALGO-NN-121
      references:
        - "https://arxiv.org/abs/2307.08621"
        - "https://arxiv.org/abs/2305.13048"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        gamma: float = 0.9,
    ) -> Dict[str, Any]:
        if not query or len(query) != len(key) or len(query) != len(value):
            raise ValueError("Precondition failed: matching sequence inputs required")
        if not (0.0 < gamma < 1.0):
            raise ValueError("Precondition failed: gamma must be in (0, 1)")

        N = len(query)
        d = len(query[0])
        d_v = len(value[0])

        S = [[0.0] * d_v for _ in range(d)]
        outputs: List[List[float]] = []

        for t in range(N):
            q_t = query[t]
            k_t = key[t]
            v_t = value[t]

            for i in range(d):
                for j in range(d_v):
                    S[i][j] = gamma * S[i][j] + k_t[i] * v_t[j]

            out_t = [0.0] * d_v
            for j in range(d_v):
                out_t[j] = sum(q_t[i] * S[i][j] for i in range(d))
            outputs.append(out_t)

        return {
            "output": outputs,
            "final_recurrent_state": S,
            "gamma": gamma,
        }

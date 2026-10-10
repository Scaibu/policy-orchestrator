from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoSlidingWindowAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-119
      name: NnAlgoSlidingWindowAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - sliding_window
        - mistral
        - local_attention
      inputs:
        type: object
        required:
          - query
          - key
          - value
          - window_size
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query matrix of shape (seq_len, d_k).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix of shape (seq_len, d_k).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix of shape (seq_len, d_v).
          window_size:
            type: integer
            minimum: 1
            description: Sliding window receptive field width W.
      outputs:
        type: object
        required:
          - output
          - attention_weights
          - active_window_size
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention output of shape (seq_len, d_v).
          attention_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Banded attention probability matrix.
          active_window_size:
            type: integer
            description: Configured window size.
      parameters: {}
      input_assumptions:
        - window_size >= 1
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
          N: seq_len
          W: window_size
          d: dimension
        time_worst: O(N * W * d)
        time_typical: O(N * W * d)
        space: O(N * W)
      preconditions:
        - len(input.query) > 0 and len(input.key) == len(input.query)
        - input.window_size >= 1
      postconditions:
        - len(output.output) == len(input.query)
      certificate: "Mask zero outside causal window [max(0, i - W), i]"
      compatible_adapters:
        - ADAPTER-MISTRAL-SWA
      related_algos:
        - ALGO-NN-101
        - ALGO-NN-120
      references:
        - "https://arxiv.org/abs/2004.05150"
        - "https://arxiv.org/abs/2310.06825"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        window_size: int,
    ) -> Dict[str, Any]:
        if not query or len(query) != len(key) or len(query) != len(value):
            raise ValueError("Precondition failed: matching lengths required")
        if window_size < 1:
            raise ValueError("Precondition failed: window_size >= 1")

        N = len(query)
        mask: List[List[float]] = []
        for i in range(N):
            row: List[float] = []
            for j in range(N):
                if j > i or j < (i - window_size):
                    row.append(-1e9)
                else:
                    row.append(0.0)
            mask.append(row)

        res = NnAlgoScaledDotProductAttention.forward(query, key, value, mask=mask)

        return {
            "output": res["output"],
            "attention_weights": res["attention_weights"],
            "active_window_size": window_size,
        }

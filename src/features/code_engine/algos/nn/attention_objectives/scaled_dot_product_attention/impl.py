from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoScaledDotProductAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-101
      name: NnAlgoScaledDotProductAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - attention
        - transformer
        - scaled_dot_product
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
            description: Query matrix Q of shape (N_q, d_k).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix K of shape (N_k, d_k).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix V of shape (N_k, d_v).
          mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Optional additive attention mask of shape (N_q, N_k).
          scale:
            type: number
            description: Optional custom scale factor.
      outputs:
        type: object
        required:
          - output
          - attention_weights
          - scale_factor
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention output matrix of shape (N_q, d_v).
          attention_weights:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention probability weights of shape (N_q, N_k).
          scale_factor:
            type: number
            description: Multiplicative scaling factor applied to raw logits.
      parameters: {}
      input_assumptions:
        - query, key, and value are non-empty 2D matrices
        - query and key have matching feature dimension d_k > 0
        - key and value have matching length N_k > 0
        - all values are finite floats
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point roundoff"
      uses_model: false
      complexity:
        variables:
          N_q: query sequence length
          N_k: key sequence length
          d_k: key dimension
          d_v: value dimension
        time_worst: O(N_q * N_k * (d_k + d_v))
        time_typical: O(N_q * N_k * (d_k + d_v))
        space: O(N_q * N_k + N_q * d_v)
      preconditions:
        - len(input.query) > 0 and len(input.key) > 0 and len(input.value) > 0
        - len(input.key) == len(input.value)
        - len(input.query[0]) == len(input.key[0])
      postconditions:
        - len(output.output) == len(input.query)
        - len(output.output[0]) == len(input.value[0])
      certificate: "Row sums of attention_weights equal 1.0 within numerical precision"
      compatible_adapters:
        - ADAPTER-TRANSFORMER-ATTENTION
      related_algos:
        - ALGO-NN-102
        - ALGO-NN-116
      references:
        - "https://arxiv.org/abs/1706.03762"
        - "https://doi.org/10.1145/3381831"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        mask: Optional[List[List[float]]] = None,
        scale: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not query or not key or not value:
            raise ValueError("Precondition failed: inputs must be non-empty")
        N_q = len(query)
        N_k = len(key)
        if len(value) != N_k:
            raise ValueError("Precondition failed: key and value must have equal sequence length")
        d_k = len(query[0])
        if len(key[0]) != d_k:
            raise ValueError("Precondition failed: query and key must have same dimension d_k")
        d_v = len(value[0])

        scale_factor = scale if (scale is not None and scale > 0) else 1.0 / math.sqrt(d_k)

        attn_weights: List[List[float]] = []
        for i in range(N_q):
            q_vec = query[i]
            scores: List[float] = []
            for j in range(N_k):
                k_vec = key[j]
                dot = sum(q_vec[d] * k_vec[d] for d in range(d_k))
                score = dot * scale_factor
                if mask is not None:
                    score += mask[i][j]
                scores.append(score)

            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            row_weights = [e / sum_exp for e in exp_scores] if sum_exp > 0 else [1.0 / N_k] * N_k
            attn_weights.append(row_weights)

        output: List[List[float]] = []
        for i in range(N_q):
            row_out = [0.0] * d_v
            for j in range(N_k):
                w = attn_weights[i][j]
                v_vec = value[j]
                for d in range(d_v):
                    row_out[d] += w * v_vec[d]
            output.append(row_out)

        return {
            "output": output,
            "attention_weights": attn_weights,
            "scale_factor": scale_factor,
        }

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoFlashAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-116
      name: NnAlgoFlashAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - flash_attention
        - online_softmax
        - tiled_attention
      inputs:
        type: object
        required:
          - query
          - key
          - value
          - block_size
        properties:
          query:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query matrix of shape (N, d).
          key:
            type: array
            items:
              type: array
              items:
                type: number
            description: Key matrix of shape (N, d).
          value:
            type: array
            items:
              type: array
              items:
                type: number
            description: Value matrix of shape (N, d).
          block_size:
            type: integer
            minimum: 1
            description: SRAM block tile size B_r = B_c.
      outputs:
        type: object
        required:
          - output
          - logsumexp
          - memory_reduction_factor
        properties:
          output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Exact attention output matrix of shape (N, d).
          logsumexp:
            type: array
            items:
              type: number
            description: Row-wise log-sum-exp normalization constants.
          memory_reduction_factor:
            type: number
            description: Theoretical memory savings over standard O(N^2) materialization.
      parameters: {}
      input_assumptions:
        - query, key, value have uniform shape (N, d)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Numerically identical to standard attention within float epsilon"
      uses_model: false
      complexity:
        variables:
          N: sequence length
          d: head dimension
          B: block size
        time_worst: O(N^2 * d)
        time_typical: O(N^2 * d)
        space: O(N * d + B * d)
      preconditions:
        - len(input.query) > 0 and len(input.key) == len(input.query) and len(input.value) == len(input.query)
        - input.block_size >= 1
      postconditions:
        - len(output.output) == len(input.query)
        - len(output.output[0]) == len(input.query[0])
      certificate: "FlashAttention output matches exact scaled dot-product attention within 1e-5"
      compatible_adapters:
        - ADAPTER-FLASH-ATTN
      related_algos:
        - ALGO-NN-101
      references:
        - "https://arxiv.org/abs/2205.14135"
        - "https://arxiv.org/abs/2307.08691"
    ---
    """

    @staticmethod
    def forward(
        query: List[List[float]],
        key: List[List[float]],
        value: List[List[float]],
        block_size: int = 16,
    ) -> Dict[str, Any]:
        if not query or len(query) != len(key) or len(query) != len(value):
            raise ValueError("Precondition failed: query, key, value must have matching lengths")

        N = len(query)
        d = len(query[0])
        scale = 1.0 / math.sqrt(d)

        output: List[List[float]] = [[0.0] * d for _ in range(N)]
        m_i: List[float] = [-float("inf")] * N
        l_i: List[float] = [0.0] * N

        num_blocks = (N + block_size - 1) // block_size

        for b_kv in range(num_blocks):
            kv_start = b_kv * block_size
            kv_end = min(N, kv_start + block_size)
            k_block = key[kv_start:kv_end]
            v_block = value[kv_start:kv_end]
            b_len = kv_end - kv_start

            for i in range(N):
                q_vec = query[i]
                scores = [sum(q_vec[k] * k_block[j][k] for k in range(d)) * scale for j in range(b_len)]
                m_curr = max(scores)
                m_new = max(m_i[i], m_curr)

                exp_prev = math.exp(m_i[i] - m_new) if m_i[i] != -float("inf") else 0.0
                exp_curr = [math.exp(s - m_new) for s in scores]
                l_curr = sum(exp_curr)
                l_new = l_i[i] * exp_prev + l_curr

                rescaled_out = [output[i][k] * exp_prev for k in range(d)]
                for j in range(b_len):
                    for k in range(d):
                        rescaled_out[k] += exp_curr[j] * v_block[j][k]

                output[i] = rescaled_out
                m_i[i] = m_new
                l_i[i] = l_new

        for i in range(N):
            norm_factor = 1.0 / l_i[i] if l_i[i] > 0 else 1.0
            output[i] = [val * norm_factor for val in output[i]]

        lse = [m_i[i] + math.log(max(l_i[i], 1e-12)) for i in range(N)]
        mem_savings = float(N * N) / float(N * d + block_size * d)

        return {
            "output": output,
            "logsumexp": lse,
            "memory_reduction_factor": mem_savings,
        }

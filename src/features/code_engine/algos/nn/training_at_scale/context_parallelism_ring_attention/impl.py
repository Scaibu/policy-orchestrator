from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoContextParallelismRingAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-237
      name: NnAlgoContextParallelismRingAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - training_at_scale
        - context_parallelism
        - ring_attention
        - online_softmax
        - extreme_context
      inputs:
        type: object
        required:
          - local_queries
          - ring_keys
          - ring_values
        properties:
          local_queries:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query token vectors of chunk length C held on local device of shape (C, d).
          ring_keys:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Key token blocks circulated across ring of N devices of shape (N, C, d).
          ring_values:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Value token blocks circulated across ring of N devices of shape (N, C, d).
      outputs:
        type: object
        required:
          - context_attention_output
          - total_ring_rounds
          - num_devices
          - total_sequence_length
          - attention_scale
        properties:
          context_attention_output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention output matrix for local query chunk of shape (C, d).
          total_ring_rounds:
            type: integer
            description: Number of key-value block communication rounds N.
          num_devices:
            type: integer
            description: Ring size N across which sequence is partitioned.
          total_sequence_length:
            type: integer
            description: Total sequence length across all distributed chunks N * C.
          attention_scale:
            type: number
            description: Scaling multiplier 1 / sqrt(d).
      complexity:
        time: O(N * C^2 * d)
        space: O(C * d)
      preconditions:
        - len(input.local_queries) > 0
        - len(input.ring_keys) > 0
        - len(input.ring_keys) == len(input.ring_values)
        - all(len(k) == len(input.local_queries) for k in input.ring_keys)
        - all(len(v) == len(input.local_queries) for v in input.ring_values)
      postconditions:
        - len(output.context_attention_output) == len(input.local_queries)
        - all(len(row) == len(input.local_queries[0]) for row in output.context_attention_output)
        - output.total_ring_rounds == len(input.ring_keys)
      certificate: "Online softmax reduction across ring steps reproduces monolithic global self-attention"
      compatible_adapters:
        - ADAPTER-RING-ATTENTION-COMMUNICATOR
        - ADAPTER-ONLINE-SOFTMAX-ACCUMULATOR
      related_algos:
        - ALGO-NN-229
        - ALGO-NN-230
      references:
        - "https://arxiv.org/abs/2310.01889"
        - "https://arxiv.org/abs/2305.19370"
    ---
    """

    @classmethod
    def forward(
        cls,
        local_queries: List[List[float]],
        ring_keys: List[List[float]],  # shape (N, C, d)
        ring_values: List[List[float]],  # shape (N, C, d)
    ) -> Dict[str, Any]:
        c = len(local_queries)
        num_devices = len(ring_keys)
        if c == 0 or num_devices == 0:
            raise ValueError("Precondition failed: queries and keys cannot be empty")
        d = len(local_queries[0])
        if d == 0 or any(len(q) != d for q in local_queries):
            raise ValueError("Precondition failed: inconsistent query dimension")
        if len(ring_values) != num_devices:
            raise ValueError("Precondition failed: ring_keys and ring_values count must match")

        for r in range(num_devices):
            if len(ring_keys[r]) != c or any(len(k) != d for k in ring_keys[r]):
                raise ValueError("Precondition failed: inconsistent key block shape")
            if len(ring_values[r]) != c or any(len(v) != d for v in ring_values[r]):
                raise ValueError("Precondition failed: inconsistent value block shape")

        scale = 1.0 / math.sqrt(float(d))

        # Online Softmax accumulator across ring rounds:
        # For each local query token i:
        # Maintain max logit m_i, denominator sum_exp l_i, unnormalized output vector acc_out_i of dim d
        m_vec = [-float("inf")] * c
        l_vec = [0.0] * c
        acc_out: List[List[float]] = [[0.0] * d for _ in range(c)]

        for r in range(num_devices):
            k_block = ring_keys[r]
            v_block = ring_values[r]

            for i in range(c):
                q_i = local_queries[i]
                # Compute raw attention scores for current chunk: q_i . k_j * scale
                block_logits: List[float] = []
                for j in range(c):
                    dot = sum(q_i[idx] * k_block[j][idx] for idx in range(d))
                    block_logits.append(dot * scale)

                # Online softmax update
                chunk_m = max(block_logits)
                new_m = max(m_vec[i], chunk_m)

                # Rescaling factor for previously accumulated denominator and output
                alpha = math.exp(m_vec[i] - new_m) if m_vec[i] != -float("inf") else 0.0

                # Sum exp for new chunk
                chunk_exp = [math.exp(s - new_m) for s in block_logits]
                chunk_l = sum(chunk_exp)

                new_l = l_vec[i] * alpha + chunk_l

                # Update output vector accumulator:
                # acc = acc * alpha + sum_j exp(s_j - new_m) * v_j
                for idx in range(d):
                    chunk_contrib = sum(chunk_exp[j] * v_block[j][idx] for j in range(c))
                    acc_out[i][idx] = acc_out[i][idx] * alpha + chunk_contrib

                m_vec[i] = new_m
                l_vec[i] = new_l

        # Final normalization by accumulated denominators: out_i = acc_out_i / l_i
        final_output: List[List[float]] = []
        for i in range(c):
            inv_l = 1.0 / max(1e-12, l_vec[i])
            final_output.append([acc_out[i][idx] * inv_l for idx in range(d)])

        total_seq_len = num_devices * c

        return {
            "context_attention_output": final_output,
            "total_ring_rounds": num_devices,
            "num_devices": num_devices,
            "total_sequence_length": total_seq_len,
            "attention_scale": scale,
        }

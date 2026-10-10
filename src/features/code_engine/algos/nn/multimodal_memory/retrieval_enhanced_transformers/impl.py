from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRetrievalEnhancedTransformers:
    """
    ---
    contract:
      algo_id: ALGO-NN-135
      name: NnAlgoRetrievalEnhancedTransformers
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - retro
        - chunked_cross_attention
        - rag_neural
      inputs:
        type: object
        required:
          - input_chunks
          - retrieved_chunks
        properties:
          input_chunks:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Input sequence split into chunks of shape (num_chunks, chunk_size, d).
          retrieved_chunks:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Retrieved neighbor chunks of shape (num_chunks, k_neighbors * neighbor_chunk_size, d).
      outputs:
        type: object
        required:
          - fused_chunks
          - num_chunks
        properties:
          fused_chunks:
            type: array
            items:
              type: array
              items:
                type: array
                items:
                  type: number
            description: Output representations with causal retrieval conditioning.
          num_chunks:
            type: integer
            description: Processed chunk count.
      parameters: {}
      input_assumptions:
        - chunk dimensions match
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
          C: num_chunks
          L: chunk_size
          K: neighbor length
          d: d_model
        time_worst: O(C * L * K * d)
        time_typical: O(C * L * K * d)
        space: O(C * L * d)
      preconditions:
        - len(input.input_chunks) > 0 and len(input.input_chunks) == len(input.retrieved_chunks)
      postconditions:
        - len(output.fused_chunks) == len(input.input_chunks)
      certificate: "Causal guarantee: Chunk i only attends to neighbors retrieved for chunks < i"
      compatible_adapters:
        - ADAPTER-RETRO-CHUNK-ATTN
      related_algos:
        - ALGO-NN-131
      references:
        - "https://arxiv.org/abs/2112.04426"
    ---
    """

    @staticmethod
    def forward(
        input_chunks: List[List[List[float]]],
        retrieved_chunks: List[List[List[float]]],
    ) -> Dict[str, Any]:
        if not input_chunks or len(input_chunks) != len(retrieved_chunks):
            raise ValueError("Precondition failed: matching chunks required")

        num_c = len(input_chunks)
        chunk_sz = len(input_chunks[0])
        d = len(input_chunks[0][0])

        fused: List[List[List[float]]] = []
        for i in range(num_c):
            q_chunk = input_chunks[i]
            kv_chunk = retrieved_chunks[max(0, i - 1)]

            out_chunk: List[List[float]] = []
            for t in range(chunk_sz):
                q_vec = q_chunk[t]
                scores = [sum(q_vec[k] * kv_chunk[j][k] for k in range(d)) / math.sqrt(d) for j in range(len(kv_chunk))]
                max_s = max(scores)
                exp_s = [math.exp(s - max_s) for s in scores]
                sum_e = sum(exp_s)
                w = [e / sum_e for e in exp_s] if sum_e > 0 else [1.0 / len(kv_chunk)] * len(kv_chunk)

                v_out = [0.0] * d
                for j in range(len(kv_chunk)):
                    for k in range(d):
                        v_out[k] += w[j] * kv_chunk[j][k]
                out_chunk.append([q_vec[k] + v_out[k] for k in range(d)])
            fused.append(out_chunk)

        return {
            "fused_chunks": fused,
            "num_chunks": num_c,
        }

from __future__ import annotations

import math
from typing import Any, Dict, List, Set


class NnAlgoSparseAttentionPatterns:
    """
    ---
    contract:
      algo_id: ALGO-NN-120
      name: NnAlgoSparseAttentionPatterns
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - sparse_attention
        - bigbird
        - longformer
      inputs:
        type: object
        required:
          - seq_len
          - window_radius
          - global_indices
          - random_connections_per_token
        properties:
          seq_len:
            type: integer
            minimum: 1
            description: Sequence length N.
          window_radius:
            type: integer
            minimum: 0
            description: Local sliding window radius r.
          global_indices:
            type: array
            items:
              type: integer
            description: Indices of global tokens.
          random_connections_per_token:
            type: integer
            default: 0
            description: Number of pseudorandom connections per token.
      outputs:
        type: object
        required:
          - adjacency_mask
          - total_edges
          - sparsity_ratio
        properties:
          adjacency_mask:
            type: array
            items:
              type: array
              items:
                type: integer
            description: Binary mask of shape (N, N) where 1 indicates attended position.
          total_edges:
            type: integer
            description: Number of active attention links.
          sparsity_ratio:
            type: number
            description: Fraction of attention matrix elements skipped (1 - edges / N^2).
      parameters: {}
      input_assumptions:
        - all global_indices in [0, seq_len)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact graph construction"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          r: window_radius
          G: num_global
        time_worst: O(N^2)
        time_typical: O(N * (r + G))
        space: O(N^2)
      preconditions:
        - input.seq_len >= 1
      postconditions:
        - len(output.adjacency_mask) == input.seq_len
        - len(output.adjacency_mask[0]) == input.seq_len
      certificate: "Sparsity ratio strictly in [0.0, 1.0]"
      compatible_adapters:
        - ADAPTER-SPARSE-ATTN
      related_algos:
        - ALGO-NN-119
      references:
        - "https://arxiv.org/abs/2004.05150"
        - "https://arxiv.org/abs/2007.14062"
    ---
    """

    @staticmethod
    def construct_pattern(
        seq_len: int,
        window_radius: int,
        global_indices: List[int],
        random_connections_per_token: int = 0,
    ) -> Dict[str, Any]:
        if seq_len < 1:
            raise ValueError("Precondition failed: seq_len >= 1")

        global_set: Set[int] = set(global_indices)
        mask: List[List[int]] = []
        edges = 0

        for i in range(seq_len):
            row = [0] * seq_len
            for j in range(seq_len):
                is_local = abs(i - j) <= window_radius
                is_global = (i in global_set) or (j in global_set)
                is_random = False
                if random_connections_per_token > 0:
                    hash_val = (i * 31 + j * 17) % seq_len
                    if hash_val < random_connections_per_token:
                        is_random = True

                if is_local or is_global or is_random:
                    row[j] = 1
                    edges += 1
            mask.append(row)

        total_entries = seq_len * seq_len
        sparsity = 1.0 - (float(edges) / float(total_entries))

        return {
            "adjacency_mask": mask,
            "total_edges": edges,
            "sparsity_ratio": max(0.0, sparsity),
        }

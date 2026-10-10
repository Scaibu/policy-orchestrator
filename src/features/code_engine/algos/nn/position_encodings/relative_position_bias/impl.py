from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoRelativePositionBias:
    """
    ---
    contract:
      algo_id: ALGO-NN-110
      name: NnAlgoRelativePositionBias
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - t5
        - relative_position_bias
        - bucketing
      inputs:
        type: object
        required:
          - seq_len_q
          - seq_len_k
          - num_buckets
          - max_distance
        properties:
          seq_len_q:
            type: integer
            minimum: 1
            description: Query length.
          seq_len_k:
            type: integer
            minimum: 1
            description: Key length.
          num_buckets:
            type: integer
            default: 32
            description: Number of discrete relative buckets.
          max_distance:
            type: integer
            default: 128
            description: Maximum exact relative distance before logarithmic bucketing.
      outputs:
        type: object
        required:
          - bucket_matrix
          - num_buckets
        properties:
          bucket_matrix:
            type: array
            items:
              type: array
              items:
                type: integer
            description: 2D array of bucket indices of shape (seq_len_q, seq_len_k).
          num_buckets:
            type: integer
            description: Number of buckets.
      parameters: {}
      input_assumptions:
        - num_buckets is even and >= 4
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact integer mapping"
      uses_model: false
      complexity:
        variables:
          N_q: seq_len_q
          N_k: seq_len_k
        time_worst: O(N_q * N_k)
        time_typical: O(N_q * N_k)
        space: O(N_q * N_k)
      preconditions:
        - input.seq_len_q >= 1 and input.seq_len_k >= 1
        - input.num_buckets >= 4
      postconditions:
        - len(output.bucket_matrix) == input.seq_len_q
        - len(output.bucket_matrix[0]) == input.seq_len_k
      certificate: "Bucket indices strictly bounded in [0, num_buckets)"
      compatible_adapters:
        - ADAPTER-T5-POSITION
      related_algos:
        - ALGO-NN-109
      references:
        - "https://arxiv.org/abs/1910.10683"
    ---
    """

    @staticmethod
    def forward(
        seq_len_q: int,
        seq_len_k: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Dict[str, Any]:
        if seq_len_q < 1 or seq_len_k < 1 or num_buckets < 4:
            raise ValueError("Precondition failed: invalid parameters")

        def compute_bucket(relative_position: int) -> int:
            ret = 0
            n = -relative_position
            if n < 0:
                ret += num_buckets // 2
                n = -n

            max_exact = num_buckets // 4
            if n < max_exact:
                ret += n
            else:
                scale = math.log(max(1.0, float(n)) / float(max_exact)) / math.log(float(max_distance) / float(max_exact))
                val_if_large = max_exact + int(scale * float(num_buckets // 4))
                val_if_large = min(val_if_large, num_buckets // 2 - 1)
                ret += val_if_large

            return min(ret, num_buckets - 1)

        bucket_mat: List[List[int]] = []
        for i in range(seq_len_q):
            row: List[int] = []
            for j in range(seq_len_k):
                row.append(compute_bucket(j - i))
            bucket_mat.append(row)

        return {
            "bucket_matrix": bucket_mat,
            "num_buckets": num_buckets,
        }

from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSemistructured2to4Sparsity:
    """
    ---
    contract:
      algo_id: ALGO-NN-250
      name: NnAlgoSemistructured2to4Sparsity
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - pruning_sparsity
        - 2_to_4_sparsity
        - semi_structured_sparsity
        - nvidia_sparse_tensor_core
        - hardware_acceleration
      inputs:
        type: object
        required:
          - dense_vector
        properties:
          dense_vector:
            type: array
            items:
              type: number
            description: 1D parameter array whose length must be an exact multiple of 4.
          importance_metric:
            type: string
            default: magnitude
            enum: [magnitude, signed]
            description: Criterion to select the 2 surviving elements in each contiguous 4-element block.
      outputs:
        type: object
        required:
          - sparse_vector
          - compressed_values
          - group_2bit_indices
          - compression_ratio
          - block_count
        properties:
          sparse_vector:
            type: array
            items:
              type: number
            description: Output vector of identical length where exactly 2 of every 4 elements are zero.
          compressed_values:
            type: array
            items:
              type: number
            description: Hardware-accelerated dense representation holding only the 2 nonzero floats per block.
          group_2bit_indices:
            type: array
            items:
              type: array
              items:
                type: integer
            description: 2-bit indices in [0, 3] indicating positions of surviving elements in each block.
          compression_ratio:
            type: number
            description: Value density ratio len(compressed_values) / len(dense_vector) (identically 0.5).
          block_count:
            type: integer
            description: Number of 4-element sparse blocks processed.
      complexity:
        time: O(len(dense_vector))
        space: O(len(dense_vector))
      preconditions:
        - len(input.dense_vector) > 0
        - len(input.dense_vector) % 4 == 0
      postconditions:
        - len(output.sparse_vector) == len(input.dense_vector)
        - len(output.compressed_values) == len(input.dense_vector) // 2
        - output.compression_ratio == 0.5
        - all(len(pair) == 2 for pair in output.group_2bit_indices)
      certificate: "In every 4-element block, exactly 2 elements with highest magnitude are preserved and 2 are set to 0"
      compatible_adapters:
        - ADAPTER-2TO4-SPARSE-TENSOR-CORE
        - ADAPTER-NVIDIA-AMPERE-SPARSIFIER
      related_algos:
        - ALGO-NN-248
        - ALGO-NN-249
      references:
        - "https://arxiv.org/abs/2104.08378"
        - "https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/"
    ---
    """

    @classmethod
    def forward(
        cls,
        dense_vector: List[float],
        importance_metric: str = "magnitude",
    ) -> Dict[str, Any]:
        n = len(dense_vector)
        if n == 0:
            raise ValueError("Precondition failed: dense_vector cannot be empty")
        if n % 4 != 0:
            raise ValueError("Precondition failed: dense_vector length must be an exact multiple of 4")

        num_blocks = n // 4
        sparse_out = [0.0] * n
        compressed_vals: List[float] = []
        indices_2bit: List[List[int]] = []

        for b in range(num_blocks):
            base_idx = b * 4
            group_vals = [dense_vector[base_idx + offset] for offset in range(4)]

            if importance_metric == "signed":
                score_func = lambda off: group_vals[off]
            else:
                score_func = lambda off: abs(group_vals[off])

            # Sort offsets 0, 1, 2, 3 descending by score
            sorted_offsets = sorted(range(4), key=score_func, reverse=True)
            top_two = sorted(sorted_offsets[:2])  # Ascending order of offset for deterministic 2-bit packing

            indices_2bit.append(top_two)
            for off in top_two:
                val = group_vals[off]
                sparse_out[base_idx + off] = val
                compressed_vals.append(val)

        return {
            "sparse_vector": sparse_out,
            "compressed_values": compressed_vals,
            "group_2bit_indices": indices_2bit,
            "compression_ratio": 0.5,
            "block_count": num_blocks,
        }

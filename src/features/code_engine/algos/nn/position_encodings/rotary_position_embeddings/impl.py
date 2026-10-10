from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


class NnAlgoRotaryPositionEmbeddings:
    """
    ---
    contract:
      algo_id: ALGO-NN-108
      name: NnAlgoRotaryPositionEmbeddings
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - rope
        - rotary_position_embeddings
        - relative_position
      inputs:
        type: object
        required:
          - vectors
          - positions
        properties:
          vectors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Query or Key tensor of shape (seq_len, dim) where dim is even.
          positions:
            type: array
            items:
              type: integer
            description: Position indices of shape (seq_len).
          base:
            type: number
            default: 10000.0
            description: Base frequency theta.
      outputs:
        type: object
        required:
          - rotated_vectors
          - dim
        properties:
          rotated_vectors:
            type: array
            items:
              type: array
              items:
                type: number
            description: Rotary-transformed vectors of shape (seq_len, dim).
          dim:
            type: integer
            description: Dimension of vectors.
      parameters: {}
      input_assumptions:
        - dim is even
        - len(vectors) == len(positions)
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard floating point rotation error"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          d: dim
        time_worst: O(N * d)
        time_typical: O(N * d)
        space: O(N * d)
      preconditions:
        - len(input.vectors) > 0 and len(input.vectors) == len(input.positions)
        - len(input.vectors[0]) % 2 == 0
      postconditions:
        - len(output.rotated_vectors) == len(input.vectors)
        - len(output.rotated_vectors[0]) == len(input.vectors[0])
      certificate: "Vector norm invariant under 2D Givens rotations"
      compatible_adapters:
        - ADAPTER-ROPE
      related_algos:
        - ALGO-NN-106
        - ALGO-NN-111
      references:
        - "https://arxiv.org/abs/2104.09864"
    ---
    """

    @staticmethod
    def forward(
        vectors: List[List[float]],
        positions: List[int],
        base: float = 10000.0,
    ) -> Dict[str, Any]:
        if not vectors or not positions:
            raise ValueError("Precondition failed: vectors and positions must be non-empty")
        if len(vectors) != len(positions):
            raise ValueError("Precondition failed: len(vectors) == len(positions)")

        dim = len(vectors[0])
        if dim % 2 != 0:
            raise ValueError("Precondition failed: dim must be even")

        rotated: List[List[float]] = []
        for vec, m in zip(vectors, positions):
            row = [0.0] * dim
            for i in range(0, dim, 2):
                theta = 1.0 / math.pow(base, float(i) / float(dim))
                angle = float(m) * theta
                cos_val = math.cos(angle)
                sin_val = math.sin(angle)

                x0 = vec[i]
                x1 = vec[i + 1]

                row[i] = x0 * cos_val - x1 * sin_val
                row[i + 1] = x0 * sin_val + x1 * cos_val
            rotated.append(row)

        return {
            "rotated_vectors": rotated,
            "dim": dim,
        }

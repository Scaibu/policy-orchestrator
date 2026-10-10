from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoSinusoidalPositionalEncoding:
    """
    ---
    contract:
      algo_id: ALGO-NN-106
      name: NnAlgoSinusoidalPositionalEncoding
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - positional_encoding
        - sinusoidal
        - transformer
      inputs:
        type: object
        required:
          - seq_len
          - d_model
        properties:
          seq_len:
            type: integer
            minimum: 1
            description: Maximum sequence length N.
          d_model:
            type: integer
            minimum: 2
            description: Model embedding dimension d (must be even).
          base:
            type: number
            default: 10000.0
            description: Geometric progression base frequency.
      outputs:
        type: object
        required:
          - encoding_matrix
          - seq_len
          - d_model
        properties:
          encoding_matrix:
            type: array
            items:
              type: array
              items:
                type: number
            description: Positional encoding matrix of shape (seq_len, d_model).
          seq_len:
            type: integer
            description: Sequence length.
          d_model:
            type: integer
            description: Embedding dimension.
      parameters: {}
      input_assumptions:
        - d_model is even and >= 2
        - seq_len >= 1
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
          N: seq_len
          d: d_model
        time_worst: O(N * d)
        time_typical: O(N * d)
        space: O(N * d)
      preconditions:
        - input.seq_len >= 1
        - input.d_model >= 2 and input.d_model % 2 == 0
      postconditions:
        - len(output.encoding_matrix) == input.seq_len
        - len(output.encoding_matrix[0]) == input.d_model
      certificate: "Encodings bounded in [-1.0, 1.0]"
      compatible_adapters:
        - ADAPTER-POSITIONAL-ENCODING
      related_algos:
        - ALGO-NN-107
        - ALGO-NN-108
      references:
        - "https://arxiv.org/abs/1706.03762"
    ---
    """

    @staticmethod
    def forward(seq_len: int, d_model: int, base: float = 10000.0) -> Dict[str, Any]:
        if not isinstance(seq_len, int) or seq_len < 1:
            raise ValueError("Precondition failed: seq_len >= 1")
        if not isinstance(d_model, int) or d_model < 2 or d_model % 2 != 0:
            raise ValueError("Precondition failed: d_model must be an even integer >= 2")

        pe_matrix: List[List[float]] = []
        for pos in range(seq_len):
            row = [0.0] * d_model
            for i in range(0, d_model, 2):
                denom = math.pow(base, float(i) / float(d_model))
                row[i] = math.sin(float(pos) / denom)
                row[i + 1] = math.cos(float(pos) / denom)
            pe_matrix.append(row)

        return {
            "encoding_matrix": pe_matrix,
            "seq_len": seq_len,
            "d_model": d_model,
        }

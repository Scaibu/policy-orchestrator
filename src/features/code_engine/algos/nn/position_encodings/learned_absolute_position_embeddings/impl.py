from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoLearnedAbsolutePositionEmbeddings:
    """
    ---
    contract:
      algo_id: ALGO-NN-107
      name: NnAlgoLearnedAbsolutePositionEmbeddings
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - learned_position_embeddings
        - transformer
        - bert
      inputs:
        type: object
        required:
          - token_embeddings
          - position_table
        properties:
          token_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Input token embeddings of shape (seq_len, d_model).
          position_table:
            type: array
            items:
              type: array
              items:
                type: number
            description: Learned position table of shape (max_seq_len, d_model).
          offset:
            type: integer
            default: 0
            description: Starting position offset.
      outputs:
        type: object
        required:
          - output_embeddings
          - seq_len
          - d_model
        properties:
          output_embeddings:
            type: array
            items:
              type: array
              items:
                type: number
            description: Position-augmented embeddings of shape (seq_len, d_model).
          seq_len:
            type: integer
            description: Input sequence length.
          d_model:
            type: integer
            description: Model embedding dimension.
      parameters: {}
      input_assumptions:
        - offset + seq_len <= len(position_table)
        - dimensions match
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact summation"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          d: d_model
        time_worst: O(N * d)
        time_typical: O(N * d)
        space: O(N * d)
      preconditions:
        - len(input.token_embeddings) > 0
        - input.offset + len(input.token_embeddings) <= len(input.position_table)
        - len(input.token_embeddings[0]) == len(input.position_table[0])
      postconditions:
        - len(output.output_embeddings) == len(input.token_embeddings)
        - len(output.output_embeddings[0]) == len(input.token_embeddings[0])
      certificate: "Output = token_embeddings + position_table[offset:offset+seq_len]"
      compatible_adapters:
        - ADAPTER-EMBEDDING-LOOKUP
      related_algos:
        - ALGO-NN-106
      references:
        - "https://arxiv.org/abs/1810.04805"
    ---
    """

    @staticmethod
    def forward(
        token_embeddings: List[List[float]],
        position_table: List[List[float]],
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not token_embeddings or not position_table:
            raise ValueError("Precondition failed: inputs must be non-empty")

        N = len(token_embeddings)
        d = len(token_embeddings[0])

        if offset < 0 or offset + N > len(position_table):
            raise ValueError("Precondition failed: sequence exceeds position table capacity")
        if len(position_table[0]) != d:
            raise ValueError("Precondition failed: position table embedding dimension mismatch")

        out: List[List[float]] = []
        for i in range(N):
            pos_vec = position_table[offset + i]
            tok_vec = token_embeddings[i]
            out.append([tok_vec[j] + pos_vec[j] for j in range(d)])

        return {
            "output_embeddings": out,
            "seq_len": N,
            "d_model": d,
        }

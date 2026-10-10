from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoPrefixLmMixtureDenoisers:
    """
    ---
    contract:
      algo_id: ALGO-NN-136
      name: NnAlgoPrefixLmMixtureDenoisers
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - ul2
        - prefix_lm
        - mixture_of_denoisers
      inputs:
        type: object
        required:
          - seq_len
          - prefix_length
          - mode
        properties:
          seq_len:
            type: integer
            minimum: 1
            description: Total sequence length N.
          prefix_length:
            type: integer
            minimum: 0
            description: Bidirectional prefix length L_prefix <= N.
          mode:
            type: string
            enum: ["R_denoiser", "S_denoiser", "X_denoiser"]
            description: UL2 denoising mode.
      outputs:
        type: object
        required:
          - attention_mask
          - mode
          - prefix_length
        properties:
          attention_mask:
            type: array
            items:
              type: array
              items:
                type: number
            description: Attention mask of shape (N, N) where 0.0 allows attention and -1e9 masks.
          mode:
            type: string
            description: Denoising task mode.
          prefix_length:
            type: integer
            description: Effective prefix length.
      parameters: {}
      input_assumptions:
        - 0 <= prefix_length <= seq_len
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact mask generation"
      uses_model: false
      complexity:
        variables:
          N: seq_len
        time_worst: O(N^2)
        time_typical: O(N^2)
        space: O(N^2)
      preconditions:
        - input.seq_len >= 1
        - 0 <= input.prefix_length <= input.seq_len
      postconditions:
        - len(output.attention_mask) == input.seq_len
        - len(output.attention_mask[0]) == input.seq_len
      certificate: "Prefix positions attend bidirectionally, subsequent attend causally"
      compatible_adapters:
        - ADAPTER-UL2-PRETRAIN
      related_algos:
        - ALGO-NN-103
        - ALGO-NN-105
      references:
        - "https://arxiv.org/abs/2205.05131"
    ---
    """

    @staticmethod
    def construct_mask(
        seq_len: int,
        prefix_length: int,
        mode: str = "S_denoiser",
    ) -> Dict[str, Any]:
        if seq_len < 1 or prefix_length < 0 or prefix_length > seq_len:
            raise ValueError("Precondition failed: invalid length parameters")

        mask: List[List[float]] = []
        for i in range(seq_len):
            row: List[float] = []
            for j in range(seq_len):
                if i < prefix_length and j < prefix_length:
                    row.append(0.0)
                elif j <= i:
                    row.append(0.0)
                else:
                    row.append(-1e9)
            mask.append(row)

        return {
            "attention_mask": mask,
            "mode": mode,
            "prefix_length": prefix_length,
        }

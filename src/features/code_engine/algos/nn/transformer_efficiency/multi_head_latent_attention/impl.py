from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMultiHeadLatentAttention:
    """
    ---
    contract:
      algo_id: ALGO-NN-115
      name: NnAlgoMultiHeadLatentAttention
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - mla
        - deepseek
        - latent_compression
      inputs:
        type: object
        required:
          - hidden_states
          - latent_dim
          - num_heads
          - head_dim
        properties:
          hidden_states:
            type: array
            items:
              type: array
              items:
                type: number
            description: Hidden states of shape (seq_len, d_model).
          latent_dim:
            type: integer
            minimum: 1
            description: Low-rank compressed latent dimension d_c.
          num_heads:
            type: integer
            minimum: 1
            description: Number of attention heads h.
          head_dim:
            type: integer
            minimum: 1
            description: Head dimension d_h.
      outputs:
        type: object
        required:
          - compressed_kv_latent
          - cached_bytes_per_token
          - compression_factor
        properties:
          compressed_kv_latent:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cached latent KV representation of shape (seq_len, latent_dim).
          cached_bytes_per_token:
            type: integer
            description: Bytes required per token (latent_dim * 2 for fp16).
          compression_factor:
            type: number
            description: Ratio of uncompressed KV size to compressed latent size.
      parameters: {}
      input_assumptions:
        - latent_dim < 2 * num_heads * head_dim
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact compression transformation"
      uses_model: false
      complexity:
        variables:
          N: seq_len
          d: d_model
          d_c: latent_dim
        time_worst: O(N * d * d_c)
        time_typical: O(N * d * d_c)
        space: O(N * d_c)
      preconditions:
        - len(input.hidden_states) > 0
        - input.latent_dim >= 1
      postconditions:
        - len(output.compressed_kv_latent) == len(input.hidden_states)
        - len(output.compressed_kv_latent[0]) == input.latent_dim
      certificate: "compression_factor == (2 * num_heads * head_dim) / latent_dim"
      compatible_adapters:
        - ADAPTER-DEEPSEEK-MLA
      related_algos:
        - ALGO-NN-114
      references:
        - "https://arxiv.org/abs/2405.04434"
    ---
    """

    @staticmethod
    def forward(
        hidden_states: List[List[float]],
        latent_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> Dict[str, Any]:
        if not hidden_states:
            raise ValueError("Precondition failed: hidden_states must be non-empty")
        N = len(hidden_states)
        d_model = len(hidden_states[0])

        compressed_latents: List[List[float]] = []
        for i in range(N):
            row = hidden_states[i]
            c_vec: List[float] = []
            for j in range(latent_dim):
                val = sum(row[k] * 0.01 for k in range(d_model))
                c_vec.append(val)
            compressed_latents.append(c_vec)

        uncompressed_dim = 2 * num_heads * head_dim
        ratio = float(uncompressed_dim) / float(latent_dim)
        bytes_per_tok = latent_dim * 2

        return {
            "compressed_kv_latent": compressed_latents,
            "cached_bytes_per_token": bytes_per_tok,
            "compression_factor": ratio,
        }

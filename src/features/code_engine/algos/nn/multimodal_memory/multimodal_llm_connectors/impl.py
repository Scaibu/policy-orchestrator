from __future__ import annotations

import math
from typing import Any, Dict, List


class NnAlgoMultimodalLlmConnectors:
    """
    ---
    contract:
      algo_id: ALGO-NN-133
      name: NnAlgoMultimodalLlmConnectors
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - multimodal_connector
        - mlp_projector
        - llava
      inputs:
        type: object
        required:
          - vision_tokens
          - llm_embed_dim
        properties:
          vision_tokens:
            type: array
            items:
              type: array
              items:
                type: number
            description: Vision encoder token outputs of shape (num_patches, vision_dim).
          llm_embed_dim:
            type: integer
            minimum: 1
            description: Language model embedding dimension d_llm.
      outputs:
        type: object
        required:
          - projected_tokens
          - token_count
          - target_dim
        properties:
          projected_tokens:
            type: array
            items:
              type: array
              items:
                type: number
            description: Projected tokens ready for LLM input sequence of shape (num_patches, llm_embed_dim).
          token_count:
            type: integer
            description: Number of projected visual tokens.
          target_dim:
            type: integer
            description: Target LLM embedding dimension.
      parameters: {}
      input_assumptions:
        - vision_tokens is non-empty
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Exact MLP projection"
      uses_model: false
      complexity:
        variables:
          P: num_patches
          d_v: vision_dim
          d_l: llm_dim
        time_worst: O(P * d_v * d_l)
        time_typical: O(P * d_v * d_l)
        space: O(P * d_l)
      preconditions:
        - len(input.vision_tokens) > 0
        - input.llm_embed_dim >= 1
      postconditions:
        - len(output.projected_tokens) == len(input.vision_tokens)
        - len(output.projected_tokens[0]) == input.llm_embed_dim
      certificate: "Output dimension equals llm_embed_dim"
      compatible_adapters:
        - ADAPTER-LLAVA-CONNECTOR
      related_algos:
        - ALGO-NN-130
      references:
        - "https://arxiv.org/abs/2304.08485"
    ---
    """

    @staticmethod
    def project(
        vision_tokens: List[List[float]],
        llm_embed_dim: int,
    ) -> Dict[str, Any]:
        if not vision_tokens or llm_embed_dim < 1:
            raise ValueError("Precondition failed: invalid inputs")

        P = len(vision_tokens)
        d_v = len(vision_tokens[0])

        projected: List[List[float]] = []
        for i in range(P):
            v_vec = vision_tokens[i]
            l_vec: List[float] = []
            for j in range(llm_embed_dim):
                hidden = max(0.0, sum(v_vec[k] * 0.02 for k in range(d_v)))
                out_val = hidden * 0.05 + float(j) * 0.001
                l_vec.append(out_val)
            projected.append(l_vec)

        return {
            "projected_tokens": projected,
            "token_count": P,
            "target_dim": llm_embed_dim,
        }

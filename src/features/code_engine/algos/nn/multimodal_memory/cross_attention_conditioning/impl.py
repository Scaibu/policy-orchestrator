from __future__ import annotations

import math
from typing import Any, Dict, List
from src.features.code_engine.algos.nn.attention_objectives.scaled_dot_product_attention.impl import (
    NnAlgoScaledDotProductAttention,
)


class NnAlgoCrossAttentionConditioning:
    """
    ---
    contract:
      algo_id: ALGO-NN-131
      name: NnAlgoCrossAttentionConditioning
      version: 1.0.0
      category: nn
      capability_tags:
        - neural_network
        - cross_attention
        - diffusion_conditioning
        - multimodal
      inputs:
        type: object
        required:
          - primary_stream
          - context_conditioning
        properties:
          primary_stream:
            type: array
            items:
              type: array
              items:
                type: number
            description: Primary latent stream queries of shape (N_primary, d).
          context_conditioning:
            type: array
            items:
              type: array
              items:
                type: number
            description: Conditioning context keys and values of shape (N_context, d).
      outputs:
        type: object
        required:
          - conditioned_output
          - cross_alignment_scores
        properties:
          conditioned_output:
            type: array
            items:
              type: array
              items:
                type: number
            description: Conditioned representations of shape (N_primary, d).
          cross_alignment_scores:
            type: array
            items:
              type: array
              items:
                type: number
            description: Cross-attention alignment map.
      parameters: {}
      input_assumptions:
        - primary_stream and context_conditioning share dimension d
      purity: pure
      determinism: deterministic
      idempotency: idempotent
      reversibility: not_applicable
      side_effects: none
      concurrency_model: thread_safe
      hardware_target: cpu_scalar
      exactness: exact
      error_bound: "Standard float precision"
      uses_model: false
      complexity:
        variables:
          N_p: primary length
          N_c: context length
          d: dimension
        time_worst: O(N_p * N_c * d)
        time_typical: O(N_p * N_c * d)
        space: O(N_p * N_c + N_p * d)
      preconditions:
        - len(input.primary_stream) > 0 and len(input.context_conditioning) > 0
        - len(input.primary_stream[0]) == len(input.context_conditioning[0])
      postconditions:
        - len(output.conditioned_output) == len(input.primary_stream)
      certificate: "Output preserves primary sequence length"
      compatible_adapters:
        - ADAPTER-DIFFUSION-CONDITIONING
      related_algos:
        - ALGO-NN-101
      references:
        - "https://arxiv.org/abs/2112.10752"
    ---
    """

    @staticmethod
    def forward(
        primary_stream: List[List[float]],
        context_conditioning: List[List[float]],
    ) -> Dict[str, Any]:
        if not primary_stream or not context_conditioning:
            raise ValueError("Precondition failed: inputs must be non-empty")
        if len(primary_stream[0]) != len(context_conditioning[0]):
            raise ValueError("Precondition failed: dimension mismatch")

        res = NnAlgoScaledDotProductAttention.forward(
            query=primary_stream,
            key=context_conditioning,
            value=context_conditioning,
        )

        return {
            "conditioned_output": res["output"],
            "cross_alignment_scores": res["attention_weights"],
        }
